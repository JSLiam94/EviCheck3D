"""Token-boundary policy and fixed paired plan. No tokenizer import or invocation
at import. encode/tokenizer are provided only inside a separately admitted job.
"""
from __future__ import annotations
import copy
import hashlib
from pref_inputs import need, semantic_pure_deletion, validate_pref_binding
from center_decision_objective_v2 import derive_shared_action_index

INAPPLICABLE = ('PURE_DELETION', 'INSERTION_OR_EMPTY_SOURCE',
                'IDENTICAL_CONTENT', 'TOKEN_BOUNDARY_UNSCORABLE')


def encode(tokenizer, target):
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    need(type(ids) is list and len(ids) == len(offsets) and ids,
         'TARGET_OFFSETS_CONTRACT')
    need(all(type(x) is int and x >= 0 for x in ids), 'TOKEN_IDS')
    need(all(len(o) == 2 and type(o[0]) is int and type(o[1]) is int and
             0 <= o[0] <= o[1] <= len(target) for o in offsets), 'TOKEN_OFFSETS')
    need(tokenizer.decode(ids, skip_special_tokens=False,
                          clean_up_tokenization_spaces=False) == target,
         'FULL_TARGET_TOKEN_ROUNDTRIP')
    eos = tokenizer.eos_token_id
    need(type(eos) is int and eos >= 0 and eos not in ids, 'TARGET_EOS_CONTRACT')
    return {'ids': ids, 'offsets': [list(o) for o in offsets]}


def content_indices(tokenizer, target, layout, interval):
    start, end = interval
    if start == end:
        return None
    selected = [i for i, (a, b) in enumerate(layout['offsets'])
                if start <= a < b <= end]
    covered = start
    for i in selected:
        a, b = layout['offsets'][i]
        if a > covered:
            return None
        covered = max(covered, b)
    if not selected or covered != end:
        return None
    content = tokenizer.decode([layout['ids'][i] for i in selected],
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if content != target[start:end]:
        return None
    return selected


def compile_row(row, tokenizer, prompt_length):
    target = row['target']
    layout = encode(tokenizer, target)
    need(prompt_length + len(layout['ids']) + 1 <= 32768, 'TRAIN_CONTEXT_OVERFLOW')
    overlap = [any(a < y and b > x for x, y in row['new_value_ranges'])
               for a, b in layout['offsets']]
    pure_deletion = row.get('pure_deletion_common_boundary', False)
    need(type(pure_deletion) is bool and pure_deletion ==
         semantic_pure_deletion(row['edits']),
         'PURE_DELETION_LAYOUT_BINDING')
    need(row['primary_type'] == 0 or any(overlap) or pure_deletion,
         'NONDELETION_REPLACEMENT_MASK_EMPTY')
    return dict(ids=layout['ids'], offsets=layout['offsets'], replacement_mask=overlap,
                pure_deletion=pure_deletion, prompt_length=prompt_length,
                complete_length=prompt_length + len(layout['ids']) + 1)


def compile_fit(model, rows, guard):
    fit = [r for r in rows if r['internal_split'] == 'fit']
    errors = [r for r in fit if r['primary_type'] != 0]
    normals = [r for r in fit if r['primary_type'] == 0]
    need(len(errors) == len(normals) == 32, 'FIT64_ONLY')
    tok = model.tokenizer
    keep = tok('[]', add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    action = derive_shared_action_index(
        [tok(r['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
         for r in errors], keep)
    need(all(tok(r['target'], add_special_tokens=False)['input_ids'] +
             [tok.eos_token_id] == keep for r in normals), 'NORMAL_KEEP_TOKENIZATION')
    templates, k_counts, reasons = [], [], {x: 0 for x in INAPPLICABLE}
    lengths, lengths_squared, total_edits, eligible = 0, 0, 0, 0
    candidate_forwards = 0
    for error, normal in zip(errors, normals):
        guard()
        clean = copy.deepcopy(error)
        clean.update(report=error['reference'], target='[]', new_value_ranges=[],
                     primary_type=0, edits=[], pref_edits=[], pure_deletion_common_boundary=False)
        items = []
        for role, r in zip(('E', 'N', 'CLEAN'), (error, normal, clean)):
            prompt_length = model.prompt_token_count(r['report'])
            token_layout = compile_row(r, tok, prompt_length)
            items.append({'role': role, 'row': r, 'layout': token_layout})
            lengths += token_layout['complete_length']
            lengths_squared += token_layout['complete_length'] ** 2
        pairs, statuses = [], []
        positive_layout = items[0]['layout']
        for j, edit in enumerate(error['edits']):
            guard()
            total_edits += 1
            binding = error['pref_edits'][j]
            validate_pref_binding(error['edits'], j, binding)
            reason = (None if binding['disposition'] == 'POTENTIALLY_ELIGIBLE'
                      else binding['disposition'])
            # Use C's independently validated complete candidates and exact
            # semantic delta intervals; never score retained anchor context.
            positive = binding['positive_target']
            negative = binding['negative_target']
            need(positive == error['target'], 'POSITIVE_TARGET_BINDING')
            negative_layout = encode(tok, negative)
            common_length = positive_layout['prompt_length'] + max(
                len(positive_layout['ids']), len(negative_layout['ids'])) + 1
            need(common_length <= 32768, 'PREF_CONTEXT_OVERFLOW')
            plus = content_indices(tok, positive, positive_layout,
                                   binding['positive_new_content_range']) if reason is None else None
            minus = content_indices(tok, negative, negative_layout,
                                    binding['negative_new_content_range']) if reason is None else None
            if reason is None and (plus is None or minus is None or
                    [positive_layout['ids'][i] for i in plus] ==
                    [negative_layout['ids'][i] for i in minus]):
                reason = 'TOKEN_BOUNDARY_UNSCORABLE'
            if reason:
                reasons[reason] += 1
                statuses.append({'edit_index': j, 'status': reason})
                continue
            pairs.append({'edit_index': j, 'positive_target': positive, 'negative_target': negative,
                          'positive_ids': positive_layout['ids'], 'negative_ids': negative_layout['ids'],
                          'positive_indices': plus, 'negative_indices': minus,
                          'common_length': common_length,
                          'prompt_length': positive_layout['prompt_length']})
            statuses.append({'edit_index': j, 'status': 'ELIGIBLE'})
            eligible += 1
            candidate_forwards += 2
            lengths += 2 * common_length
            lengths_squared += 2 * common_length ** 2
        templates.append({'items': items, 'pairs': pairs, 'edit_statuses': statuses})
        k_counts.append(len(pairs))
    need(eligible + sum(reasons.values()) == total_edits, 'PREF_DENOMINATOR_CLOSURE')
    need(eligible > 0, 'NO_ELIGIBLE_PREF_NO_LAUNCH')
    # Reused-development branch: include all eligible edits; no mean-k filter.
    return {'templates': templates, 'action_spec': action,
            'safe': {'fit_reports': 64, 'fit_error_reports': 32, 'gold_edits': total_edits,
                     'eligible_edits': eligible, 'eligible_reports': sum(k > 0 for k in k_counts),
                     'inapplicable_counts': reasons, 'k_counts_histogram':
                         {str(k): k_counts.count(k) for k in sorted(set(k_counts))},
                     'sum_L_per_cycle': lengths, 'sum_L_squared_per_cycle': lengths_squared,
                     'sum_L_16cycles': lengths * 16,
                     'sum_L_squared_16cycles': lengths_squared * 16,
                     'full_decoder_forwards_16cycles': (96 + candidate_forwards) * 16,
                     'generation_slots_per_native': 160, 'completion_guarantee': False}}


def hold_donors(rows):
    """Fixed reused E28/N4 separate cyclic permutations; no output/quality access."""
    hold = [r for r in rows if r['internal_split'] == 'hold']
    donors = {}
    for is_normal in (False, True):
        ids = [r['case_id'] for r in hold if (r['primary_type'] == 0) == is_normal]
        expected = 4 if is_normal else 28
        need(len(ids) == len(set(ids)) == expected, 'REUSED_DEV28E4N_STRATUM')
        ids.sort(key=lambda cid: (hashlib.sha256(
            ('MAIN96_REUSED_PREF512_DEV_V1|' + cid).encode()).hexdigest(), cid))
        donors.update({cid: ids[(i + 1) % len(ids)] for i, cid in enumerate(ids)})
    return donors
