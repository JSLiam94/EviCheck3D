"""Reviewed MAIN96 reused-development input adapter. No tokenizer/Torch imports.
Imports perform no file or patient access. No exclusion registry is required or
fabricated: the entire cohort is explicitly historical/reused development.
"""
from __future__ import annotations
import copy
import hashlib
import io
from pathlib import Path
from pref_inputs import (need, private_bytes, strict_json, serialize, apply,
                         delta_zone, escaped_delta_range, semantic_pure_deletion)

PROJECTION_PIN = '536c6af8af4aa20ee6e3466de655a5bcb9c22fca5663015542527a4335e8eb86'
CACHE_PIN = '0772a7a65436bc7fd86804209529196e3900fc925b44ca66b5c05b299c94a7e9'
CACHE_SAFE_PIN = '5d648958095c7b3bbccd94a9d05feb514c9100307ce4b478aec3acddbd9ad0cd'
FIT_COUNTS = {0:32,1:5,2:5,3:5,4:5,5:4,6:4,7:4}
DEV_COUNTS = {i:4 for i in range(8)}


def preference_bindings(edits):
    result = []
    for i, edit in enumerate(edits):
        a, b, c = delta_zone(edit['old'], edit['new'])
        old_value, new_value = edit['old'][a:b], edit['new'][a:c]
        disposition = ('PURE_DELETION' if not new_value else
                       'INSERTION_OR_EMPTY_SOURCE' if not old_value else
                       'IDENTICAL_CONTENT' if old_value == new_value else 'POTENTIALLY_ELIGIBLE')
        positive, prange = escaped_delta_range(edits, i, a, c)
        foil = [dict(e) for e in edits]
        foil[i]['new'] = edit['old']
        negative, nrange = escaped_delta_range(foil, i, a, b)
        result.append(dict(edit_index=i, disposition=disposition,
                           positive_target=positive, negative_target=negative,
                           positive_new_content_range=prange, negative_new_content_range=nrange,
                           source_value=new_value, error_value=old_value,
                           token_eligibility='PENDING_RUNTIME_NOT_TOKENIZED'))
    return result


def adapt_rows(original):
    """Pure, no I/O. Validate original identity/strata; create fit candidates only.
    Keep source order, original primary types, case+CT metadata and all targets.
    Source provenance is retained privately but never enters inference prompts.
    """
    need(original['protocol'] == 'unit-program-main96-projection-v1', 'OLD_MAIN96_PROTOCOL')
    rows = original['rows']
    need(type(rows) is list and len(rows) == 96, 'OLD_MAIN96_FIXED96')
    need(len({r['case_id'] for r in rows}) == len({r['ct_key'] for r in rows}) == 96,
         'OLD_MAIN96_UNIQUE_CASE_CT')
    for split, counts in [('fit', FIT_COUNTS), ('dev', DEV_COUNTS)]:
        subset = [r for r in rows if r['internal_split'] == split]
        observed = {i:sum(r['primary_type'] == i for r in subset) for i in range(8)}
        need(len(subset) == sum(counts.values()) and observed == counts, 'ACTUAL_STRATA_NOT_BALANCED_DEV')
    adapted = []
    for original_row in rows:
        r = copy.deepcopy(original_row)
        need(r['target_status'] == 'READY' and type(r['primary_type']) is int
             and 0 <= r['primary_type'] <= 7, 'OLD_READY_TARGET_TYPE')
        target, ranges, _ = serialize(r['edits'])
        need(target == r['target'] and ranges == r['new_value_ranges'], 'OLD_CANONICAL_TARGET')
        need(apply(r['report'], r['edits']) == r['reference'], 'OLD_PATCH_RECONSTRUCTION')
        need((bool(r['edits']) and r['report'] != r['reference']) if r['primary_type'] else
             (not r['edits'] and r['report'] == r['reference']), 'OLD_ERROR_NORMAL_IDENTITY')
        r['internal_split'] = 'hold' if r['internal_split'] == 'dev' else 'fit'
        r['canonical_ct_id'] = original_row['ct_key']
        r['pure_deletion_common_boundary'] = semantic_pure_deletion(r['edits'])
        r['clean_twin'] = {'report': r['reference'], 'target': '[]'}
        r['token_eligibility'] = 'PENDING_RUNTIME_NOT_TOKENIZED'
        r['pref_edits'] = (preference_bindings(r['edits'])
                           if r['internal_split'] == 'fit' and r['primary_type'] else [])
        adapted.append(r)
    return {'protocol':'main96-reused-content-pref512-input-v1', 'rows':adapted,
            'data_history':'REUSED_NOT_NEW_NOT_INDEPENDENT',
            'original_projection_sha256':PROJECTION_PIN,
            'fit_errors':32,'fit_normals':32,'dev_errors':28,'dev_normals':4}


def load_projection(spec, guard):
    need(spec['projection_sha256'] == PROJECTION_PIN, 'FROZEN_OLD_MAIN96_INPUT_PIN')
    raw = private_bytes(spec['projection_path'], PROJECTION_PIN)
    original = strict_json(raw)
    guard()
    adapted = adapt_rows(original)
    # Only an accepted old fixed input is used. Do not forge fresh registry,
    # no-overlap attestations or producer/source identity as a new materializer.
    need(hashlib.sha256(private_bytes(spec['projection_path'])).hexdigest() == PROJECTION_PIN,
         'OLD_MAIN96_CHANGED')
    return adapted


def load_features(T, spec, data, guard):
    need(spec['cache_index_sha256'] == CACHE_PIN and
         spec['cache_safe_status_sha256'] == CACHE_SAFE_PIN, 'FROZEN_CACHE_PINS')
    safe = strict_json(private_bytes(spec['cache_safe_status_path'], CACHE_SAFE_PIN))
    need(safe['status_enum'] == 'COMPLETE_FROZEN_FEATURES' and safe['phase'] == 'cache'
         and safe['pair_id'] == spec['primary_pair_id']
         and safe['root_config_sha256'] == spec['primary_root_config_sha256']
         and safe['source_manifest_sha256'] == spec['primary_source_manifest_sha256']
         and safe['cache_index_global_sha256'] == CACHE_PIN
         and safe['planned_CT'] == safe['encoded_CT'] == 96 and safe['reused_CT'] == 0,
         'OLD_CACHE_ACCEPTED_PRODUCER_SAFE')
    index = strict_json(private_bytes(spec['cache_index_path'], CACHE_PIN))
    need(index['protocol'] == 'main96-frozen-CT-cache-index-v1' and index['complete'] is True
         and index['projection_sha256'] == PROJECTION_PIN and len(index['entries']) == 96,
         'OLD_COMPLETE_CACHE96')
    provenance = spec['cache_provenance']
    need(all(index[k] == v for k,v in provenance.items()) and
         index['feature_shape'] == [865,512] and index['feature_dtype'] == 'float32'
         and index['frozen_CT_only'] is True and index['LM_hidden_cached'] is False
         and index['bridge_cached'] is False, 'OLD_FEATURE_PROVENANCE')
    source = {r['case_id']:r for r in data['rows']}
    result = {}
    cache_root = Path(spec['cache_root'])
    need(cache_root.resolve(strict=True) == cache_root, 'OLD_CACHE_CANONICAL_ROOT')
    for e in index['entries']:
        guard()
        cid = e['case_id']
        need(cid in source and cid not in result and
             e['CT_input_stat_sha256'] == source[cid]['ct_stat_sha256'], 'OLD_CT_BINDING')
        path = Path(e['feature_file'])
        need(path.parent == cache_root, 'OLD_FEATURE_ROOT')
        payload = T.load(io.BytesIO(private_bytes(path, e['feature_sha256'], 8 << 20)),
                         map_location='cpu', weights_only=True)
        x = payload['feature']
        need(isinstance(x,T.Tensor) and tuple(x.shape) == (865,512) and
             x.device.type == 'cpu' and x.dtype == T.float32 and x.layout == T.strided
             and not x.requires_grad and bool(T.isfinite(x).all()), 'OLD_CACHE_FINITE_FEATURE')
        keys = ('CT_input_stat_sha256','cache_key_sha256',*provenance,
                'feature_dtype','feature_shape','frozen_CT_only','LM_hidden_cached','bridge_cached')
        need(all(payload[k] == e[k] for k in keys), 'OLD_PAYLOAD_ENTRY_BINDING')
        need(all(e[k] == v for k,v in provenance.items()), 'OLD_ENTRY_PROVENANCE')
        result[cid] = x.contiguous()
    need(set(result) == set(source), 'OLD_CACHE96_DENOMINATOR')
    return result
