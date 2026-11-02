"""Fixed report+CT-only generation slots, no DEV selection or semantic claims."""
from __future__ import annotations
import json
import os
from pathlib import Path
from pref_inputs import need
from pref_runtime import sync_directory, write_once


def evaluate(model, hold, features, scope, node, guard, *, progress, donors=None):
    need(node in (0, 128, 256, 512) and len(hold) == 32 and
         all(r['internal_split'] == 'hold' for r in hold), 'HOLD32_ONLY')
    label = 'hold_deranged' if donors is not None else 'hold_bound'
    need(donors is None or node == 512, 'FIXED_DERANGED_NODE')
    path = Path(scope) / f'{label}_{node:03d}_predictions_private.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    counts, strata = {}, {'error': {'reports': 0, 'exact_reference': 0, 'changed': 0},
                          'normal': {'reports': 0, 'exact_reference': 0, 'changed': 0}}
    model.eval()
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            for row in hold:
                guard()
                cid = donors[row['case_id']] if donors is not None else row['case_id']
                progress['generation_slots_attempted'] += 1
                try:
                    # No target/reference/type/span passed into inference.
                    predicted, audit = model.predict(row['report'], features[cid])
                except ValueError as exc:
                    if str(exc) != 'NO_GENERATION_TRUNCATION':
                        raise
                    predicted = None
                    audit = {'accepted': False, 'status': 'NA_GENERATION_CONTEXT_OVERFLOW'}
                record = {'case_id': row['case_id'], 'primary_type': row['primary_type'],
                          'input_report': row['report'], 'reference': row['reference'],
                          'target': row['target'], 'predicted_report': predicted,
                          'program_audit_private': audit}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode() + b'\n')
                stream.flush()
                progress['completed_generation_slots'] += 1
                status = audit['status']
                counts[status] = counts.get(status, 0) + 1
                group = strata['normal' if row['primary_type'] == 0 else 'error']
                group['reports'] += 1
                if predicted is not None:
                    group['exact_reference'] += int(predicted == row['reference'])
                    group['changed'] += int(predicted != row['report'])
                guard()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    sync_directory(scope)
    need(sum(counts.values()) == 32 and strata['error']['reports'] == 28 and strata['normal']['reports'] == 4,
         'GENERATION_DENOMINATOR_CLOSURE')
    result = {'node': node, 'condition': label, 'planned_slots': 32, 'completed_slots': 32,
              'status_counts': counts, 'literal_diagnostics_only': strata,
              'semantic_repair_harm_new_errors_CT_evidence': None,
              'semantic_status': 'NA_NOT_EVALUATED', 'patient_entries_exported': False}
    write_once(Path(scope) / f'{label}_{node:03d}_safe.json', result, guard)
    return result
