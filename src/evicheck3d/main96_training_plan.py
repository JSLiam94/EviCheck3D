"""Pure TRAIN-only paired-preservation schedule; importing performs no I/O.
All returned rows/identities are host-private. Source candidate, no launch API.
"""
from __future__ import annotations
import copy
import hashlib
import json

PROTOCOL = 'train-paired-preservation-v1'
ARMS = ('preserve_bound', 'preserve_deranged')
UPDATES = 256
TUPLE_SIZE = 3


def need(ok, code):
    if not ok:
        raise ValueError(code)


def _apply(report, edits):
    spans = []
    for edit in edits:
        need(type(edit) is dict and set(edit) == {'old', 'new'}, 'EDIT_SCHEMA_REFUSED')
        old, new = edit['old'], edit['new']
        need(type(old) is str and old and type(new) is str, 'EDIT_VALUE_REFUSED')
        need(report.count(old) == 1, 'EDIT_ANCHOR_NOT_UNIQUE')
        start = report.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'EDIT_OVERLAP')
    for start, end, new in reversed(spans):
        report = report[:start] + new + report[end:]
    return report


def _validated_fit(fit):
    need(type(fit) is list and len(fit) == 64, 'FIXED_TRAIN64_REQUIRED')
    ids = []
    for row in fit:
        need(type(row) is dict and row.get('internal_split') == 'fit', 'TRAIN_FIT_ONLY')
        need(row.get('target_status') == 'READY', 'READY_TARGET_REQUIRED')
        cid = row.get('case_id')
        need(type(cid) is str and cid, 'CASE_ID_REQUIRED')
        ids.append(cid)
        need(type(row.get('report')) is str and row['report'] and
             type(row.get('reference')) is str and row['reference'], 'TEXT_REQUIRED')
        need(type(row.get('primary_type')) is int and row['primary_type'] >= 0, 'TYPE_REQUIRED')
        need(type(row.get('target')) is str, 'PROGRAM_REQUIRED')
        edits = json.loads(row['target'])
        need(type(edits) is list, 'PROGRAM_ARRAY_REQUIRED')
        need(_apply(row['report'], edits) == row['reference'], 'REFERENCE_RECONSTRUCTION')
        need(edits == row.get('edits'), 'EDIT_BINDING_REQUIRED')
        if row['primary_type'] == 0:
            need(row['report'] == row['reference'] and row['target'] == '[]' and not edits,
                 'NORMAL_KEEP_REQUIRED')
        else:
            need(row['report'] != row['reference'] and edits and row.get('new_value_ranges'),
                 'NONIDENTITY_ERROR_REQUIRED')
    need(len(set(ids)) == 64, 'UNIQUE_TRAIN_IDS_REQUIRED')
    ordered = list(fit)  # Preserve frozen source order within strata.
    errors = [row for row in ordered if row['primary_type'] != 0]
    normals = [row for row in ordered if row['primary_type'] == 0]
    need(len(errors) == len(normals) == 32, 'FIXED_STRATA32_REQUIRED')
    return ordered, errors, normals


def compile_private_plan(fit, arm):
    """Exactly 32 fixed tuples x8 cycles, with no DEV/candidate selection.

    CT derangement is a +1 cyclic shift on each32 error/normal hash-sorted case IDs, independent
    of primary type. The clean and corrupted reports always share a donor.
    """
    need(arm in ARMS, 'UNKNOWN_ARM')
    ordered, errors, normals = _validated_fit(fit)
    donor = {}
    # Broad annotation class is preserved so each CT's weighted exposure
    # frequency (error twice including clean twin, normal once) is exact.
    for stratum in (errors,normals):
        donor_ids = sorted((row['case_id'] for row in stratum),
                           key=lambda cid: (hashlib.sha256(
                               f'{PROTOCOL}|20261025|{cid}'.encode()).hexdigest(), cid))
        donor.update({cid: donor_ids[(i + 1) % len(donor_ids)] for i,cid in enumerate(donor_ids)})
    templates = []
    for error, normal in zip(errors, normals):
        clean = copy.deepcopy(error)
        clean.update(report=error['reference'], target='[]', new_value_ranges=[], edits=[],
                     primary_type=0, annotation_pairs=[], supervision_pairs=[],
                     target_status='READY', refusal_enum=None)
        second = clean
        roles = ('CORRUPT_GOLD', 'ORIGINAL_NORMAL_KEEP', 'CLEAN_REFERENCE_KEEP')
        items = []
        for row, role in zip((error, normal, second), roles):
            original_id = row['case_id']
            ct_id = donor[original_id] if arm == 'preserve_deranged' else original_id
            # Explicit model boundary excludes reference, annotation pairs and CT paths.
            training_row = {key: copy.deepcopy(row[key]) for key in (
                'case_id', 'report', 'target', 'new_value_ranges', 'primary_type',
                'internal_split', 'target_status')}
            items.append(dict(role=role, row=training_row, feature_case_id=ct_id))
        templates.append(tuple(items))
    return dict(protocol=PROTOCOL, arm=arm, split='TRAIN_FIT_ONLY',
                optimizer_updates=UPDATES, exposure_count=UPDATES * TUPLE_SIZE,
                original_cases=64, tuples_per_cycle=32, cycles=8,
                donor_rule='WITHIN_ERROR32_NORMAL32_SHA256_PROTOCOL_SEED20261025_ID_SORTED_CYCLIC_PLUS_ONE' if arm == 'preserve_deranged' else 'BOUND_SELF',
                tuples=tuple(templates), requires_external_root_admission=True)


def update_items(private_plan, update):
    need(private_plan.get('protocol') == PROTOCOL and private_plan.get('split') == 'TRAIN_FIT_ONLY',
         'PLAN_PROTOCOL_REFUSED')
    need(type(update) is int and 1 <= update <= UPDATES, 'UPDATE_OUT_OF_RANGE')
    need(private_plan.get('optimizer_updates') == UPDATES and len(private_plan['tuples']) == 32,
         'SCHEDULE_REFUSED')
    return copy.deepcopy(private_plan['tuples'][(update - 1) % 32])
