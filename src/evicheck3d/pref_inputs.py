"""Private materializer -> trainer API. No I/O or neural import at import time.
The selector/materializer, not this loader, owns raw TRAIN/exclusion validation.
All returned identities/text/token details remain host-only.
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import stat

PROTOCOL = 'local-pref-train96-projection-v1'
CACHE_PROTOCOL = 'local-content-pref-train96-cache-v1'


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def reject(_):
        raise RuntimeError('NONFINITE_JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def private_bytes(path, pin=None, limit=32 << 20):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'INPUT_CANONICAL')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_nlink == 1
             and stat.S_IMODE(st.st_mode) == 0o600 and 0 < st.st_size <= limit,
             'INPUT_PRIVATE_METADATA')
        raw = bytearray()
        while len(raw) < st.st_size:
            part = os.read(fd, min(65536, st.st_size - len(raw)))
            need(part, 'INPUT_SHORT_READ')
            raw.extend(part)
        after = os.fstat(fd)
        stamp = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        need(stamp(st) == stamp(after), 'INPUT_CHANGED')
    finally:
        os.close(fd)
    raw = bytes(raw)
    if pin is not None:
        need(hashlib.sha256(raw).hexdigest() == pin, 'INPUT_SHA256')
    return raw


def serialize(edits):
    """Match pinned_unit_data.serialize_program, including quote-overlap SFT ranges."""
    text = '['
    ranges, interiors = [], []
    for i, edit in enumerate(edits):
        need(type(edit) is dict and set(edit) == {'old', 'new'}, 'EDIT_SCHEMA')
        need(type(edit['old']) is str and edit['old'] and type(edit['new']) is str,
             'EDIT_VALUES')
        if i:
            text += ','
        text += '{"old":' + json.dumps(edit['old'], ensure_ascii=False) + ',"new":'
        start = len(text)
        value = json.dumps(edit['new'], ensure_ascii=False)
        text += value
        ranges.append([start, len(text)])
        interiors.append([start + 1, len(text) - 1])
        text += '}'
    return text + ']', ranges, interiors


def delta_zone(old, new):
    """Shared C semantic rule: longest common prefix, then nonoverlapping suffix.
    Returns (prefix_end, old_delta_end, new_delta_end), in Unicode characters.
    """
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(old) - prefix, len(new) - prefix)
           and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]):
        suffix += 1
    return prefix, len(old) - suffix, len(new) - suffix


def semantic_pure_deletion(edits):
    return bool(edits) and all(delta_zone(e['old'], e['new'])[0] ==
                               delta_zone(e['old'], e['new'])[2] for e in edits)


def escaped_delta_range(edits, index, start, end):
    """Map a semantic Unicode slice to complete canonical escaped-JSON offsets.
    No tokenizer, local substring tokenization, or boundary widening is involved.
    """
    target, _, interiors = serialize(edits)
    new = edits[index]['new']
    need(type(start) is int and type(end) is int and 0 <= start <= end <= len(new),
         'SEMANTIC_DELTA_RANGE')
    left = len(json.dumps(new[:start], ensure_ascii=False)) - 2
    right = len(json.dumps(new[:end], ensure_ascii=False)) - 2
    base = interiors[index][0]
    interval = [base + left, base + right]
    need(interiors[index][0] <= interval[0] <= interval[1] <= interiors[index][1],
         'ESCAPED_DELTA_INTERIOR')
    need(json.loads('"' + target[interval[0]:interval[1]] + '"') == new[start:end],
         'ESCAPED_DELTA_ROUNDTRIP')
    return target, interval


def validate_pref_binding(edits, index, binding):
    """Validate C's changed-content fields against original canonical edits.
    source_value is corrected new_delta; error_value is original old_delta.
    Full foil.new remains original edit.old, including unchanged anchor context.
    """
    need(binding['token_eligibility'] == 'PENDING_RUNTIME_NOT_TOKENIZED',
         'PREF_TOKEN_ELIGIBILITY_PENDING_RUNTIME')
    edit = edits[index]
    a, b, c = delta_zone(edit['old'], edit['new'])
    old_value, new_value = edit['old'][a:b], edit['new'][a:c]
    if not new_value:
        disposition = 'PURE_DELETION'
    elif not old_value:
        disposition = 'INSERTION_OR_EMPTY_SOURCE'
    elif old_value == new_value:
        disposition = 'IDENTICAL_CONTENT'
    else:
        disposition = 'POTENTIALLY_ELIGIBLE'
    positive, positive_interval = escaped_delta_range(edits, index, a, c)
    source_copy = edit['new'][:a] + old_value + edit['new'][c:]
    need(source_copy == edit['old'], 'FULL_SOURCE_COPY_FOIL_BINDING')
    negative_edits = [dict(e) for e in edits]
    negative_edits[index]['new'] = source_copy
    negative, negative_interval = escaped_delta_range(
        negative_edits, index, a, a + len(old_value))
    for field in ('positive_new_content_range', 'negative_new_content_range'):
        need(type(binding[field]) is list and len(binding[field]) == 2
             and all(type(value) is int for value in binding[field]),
             'SEMANTIC_DELTA_RANGE_SCHEMA')
    need(type(binding['edit_index']) is int and binding['edit_index'] == index
         and binding['disposition'] == disposition
         and binding['source_value'] == new_value and binding['error_value'] == old_value
         and binding['positive_target'] == positive
         and binding['negative_target'] == negative
         and binding['positive_new_content_range'] == positive_interval
         and binding['negative_new_content_range'] == negative_interval,
         'SEMANTIC_DELTA_CANDIDATE_BINDING')


def apply(report, edits):
    spans = []
    for edit in edits:
        need(report.count(edit['old']) == 1, 'SOURCE_ANCHOR_NONUNIQUE')
        start = report.index(edit['old'])
        spans.append((start, start + len(edit['old']), edit['new']))
    spans.sort()
    need(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'SOURCE_EDIT_OVERLAP')
    for a, b, new in reversed(spans):
        report = report[:a] + new + report[b:]
    return report


def load_projection(spec, guard):
    raw = private_bytes(spec['projection_path'], spec['projection_sha256'])
    doc = strict_json(raw)
    need(doc['protocol'] == PROTOCOL and doc['status'] == 'READY'
         and doc['tokeneligibility'] == 'PENDING_RUNTIME'
         and doc['selection_salt'] == 'LOCAL_REPLACEMENT_PREF_TRAIN96_V1_20261025',
         'PROJECTION_PROTOCOL')
    for field in ('exclusion_registry_sha256', 'selection_manifest_sha256',
                  'materializer_source_manifest_sha256', 'targets_sha256'):
        need(doc[field] == spec[field], 'MATERIALIZER_RECEIPT_BINDING')
    need(doc['prior_case_overlap'] == doc['prior_canonical_CT_overlap'] == 0
         and doc['no_refill'] is True and doc['selection_before_target_validation'] is True
         and doc['test_calibration_old_DEV_read'] is False, 'MATERIALIZATION_POLICY')
    rows = doc['rows']
    need(type(rows) is list and len(rows) == 96, 'PLANNED96_REQUIRED')
    need(len({r['case_id'] for r in rows}) == len({r['canonical_ct_id'] for r in rows}) ==
         len({(r['source_dataset'], r['source_case_id']) for r in rows}) == 96,
         'CASE_CT96_UNIQUE')
    for r in rows:
        guard()
        need(r['token_eligibility'] == 'PENDING_RUNTIME_NOT_TOKENIZED',
             'ROW_TOKEN_ELIGIBILITY_PENDING_RUNTIME')
        need(type(r['case_id']) is str and r['case_id'] and
             type(r['canonical_ct_id']) is str and r['canonical_ct_id'], 'IDENTITY_SCHEMA')
        need(r['internal_split'] in ('fit', 'hold') and r['target_status'] == 'READY'
             and type(r['primary_type']) is int and 0 <= r['primary_type'] <= 7,
             'ROW_TARGET_SCHEMA')
        need(type(r['report']) is str and r['report'] and
             type(r['reference']) is str and r['reference'], 'REPORT_SCHEMA')
        target, ranges, _ = serialize(r['edits'])
        need(r['target'] == target and r['new_value_ranges'] == ranges,
             'CANONICAL_TARGET_RANGES')
        need(apply(r['report'], r['edits']) == r['reference'], 'PATCH_ROUNDTRIP')
        if r['primary_type'] == 0:
            need(not r['edits'] and r['report'] == r['reference'], 'NORMAL_KEEP')
        else:
            need(r['edits'] and r['report'] != r['reference'], 'ERROR_NONIDENTITY')
        need(r['clean_twin'] == {'report': r['reference'], 'target': '[]'},
             'CLEAN_TWIN_BINDING')
        if r['internal_split'] == 'hold':
            need(not r.get('pref_edits'), 'HOLD_MUST_NOT_HAVE_PREF_CANDIDATES')
        if r['internal_split'] == 'fit' and r['primary_type'] != 0:
            need(type(r.get('pure_deletion_common_boundary')) is bool
                 and r['pure_deletion_common_boundary'] == semantic_pure_deletion(r['edits']),
                 'PURE_DELETION_COMMON_BOUNDARY_BINDING')
            bindings = r['pref_edits']
            need(len(bindings) == len(r['edits']), 'SOURCE_UNIT_BINDING_COUNT')
            for i, binding in enumerate(bindings):
                validate_pref_binding(r['edits'], i, binding)
    for split, n in (('fit', 32), ('hold', 16)):
        subset = [r for r in rows if r['internal_split'] == split]
        need(len(subset) == n * 2 and
             sum(r['primary_type'] == 0 for r in subset) == n, 'FIXED_E_N_SPLIT')
    need(doc['targets_sha256'] == digest([(r['case_id'], r['target_status'], r['target'])
                                          for r in rows]), 'TARGETS_DIGEST')
    need(hashlib.sha256(private_bytes(spec['projection_path'])).hexdigest() ==
         spec['projection_sha256'], 'PROJECTION_RECHECK')
    return doc


def load_features(T, spec, doc, guard):
    """Read reviewed cached CT features only. Never encode CT or warm a cache."""
    import io
    index = strict_json(private_bytes(spec['cache_index_path'], spec['cache_index_sha256']))
    need(index['protocol'] == CACHE_PROTOCOL and index['complete'] is True and
         index['projection_sha256'] == spec['projection_sha256'], 'CACHE_PROJECTION')
    provenance = spec['cache_provenance']
    need(set(provenance) == {'asset_receipt_sha256', 'encoder_source_sha256',
                            'preprocessing_source_sha256'}, 'CACHE_PROVENANCE_SCHEMA')
    need(all(index[k] == v for k, v in provenance.items()), 'CACHE_PROVENANCE_PIN')
    need(index['feature_shape'] == [865, 512] and index['feature_dtype'] == 'float32'
         and index['frozen_CT_only'] is True and index['LM_hidden_cached'] is False
         and index['bridge_cached'] is False, 'CACHE_FEATURE_POLICY')
    rows = {r['case_id']: r for r in doc['rows']}
    entries = index['entries']
    need(len(entries) == 96, 'CACHE96_REQUIRED')
    result = {}
    allowed_root = Path(spec['cache_root']).resolve(strict=True)
    for e in entries:
        guard()
        cid = e['case_id']
        need(cid in rows and cid not in result and
             e['canonical_ct_id'] == rows[cid]['canonical_ct_id'], 'CACHE_IDENTITY')
        path = Path(e['feature_file'])
        need(path.parent == allowed_root, 'CACHE_FILE_ROOT')
        payload = T.load(io.BytesIO(private_bytes(path, e['feature_sha256'], 8 << 20)),
                         map_location='cpu', weights_only=True)
        x = payload['feature']
        need(isinstance(x, T.Tensor) and tuple(x.shape) == (865, 512) and
             x.dtype == T.float32 and x.device.type == 'cpu' and
             x.layout == T.strided and not x.requires_grad and bool(T.isfinite(x).all()),
             'CACHE_TENSOR_CONTRACT')
        for key in ('canonical_ct_id', 'cache_key_sha256', *provenance):
            need(payload[key] == e[key], 'CACHE_PAYLOAD_PROVENANCE')
        need(all(e[k] == v for k, v in provenance.items()), 'CACHE_ENTRY_PROVENANCE')
        result[cid] = x.contiguous()
    need(set(result) == set(rows), 'CACHE_CLOSED_DENOMINATOR')
    return result
