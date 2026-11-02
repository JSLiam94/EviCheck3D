"""Fixed mean3 + content preference update. Neural stack injected by caller.
A lambda=0 arm still evaluates both candidate graphs and calls joint backward.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pref_inputs import need
from main96_train_step import _diagnose_gradients


def operation_seed(train_seed, update, role, edit_index):
    need(role in ('E', 'N', 'CLEAN', 'PREF') and type(update) is int and 1 <= update <= (1<<31)-1,
         'OPERATION_ADDRESS')
    raw = json.dumps([train_seed, update, role, edit_index],
                     separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & ((1 << 63) - 1)


@contextmanager
def addressed_rng(T, device, seed):
    index = device.index
    need(type(index) is int and T.cuda.device_count() == 1 and index == 0, 'OWN_DEVICE_ONLY')
    with T.random.fork_rng(devices=[index], enabled=True):
        T.default_generator.manual_seed(seed)
        T.cuda.default_generators[index].manual_seed(seed)
        yield


def train_step(T, model, optimizer, plan, update, features, train_seed, weight, guard, progress):
    need(weight in (0.0, 0.1) and type(update) is int and 1 <= update <= (1<<31)-1, 'FIXED_TRAIN_CONFIG')
    guard()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = model.export()
    template = plan['template_at'](plan['next_pool_cursor'])
    observations, gaps = [], []
    try:
        for item in template['items']:
            guard()
            row = item['row']
            need(row['internal_split'] == 'fit', 'NO_HOLD_IN_TRAINING')
            seed = operation_seed(train_seed, update, item['role'], -1)
            with addressed_rng(T, model.device, seed):
                with T.autocast('cuda', dtype=T.bfloat16):
                    loss, metrics = model.sft_loss(row, features[row['case_id']],
                                                  item['layout'], plan['action_spec']['action_index'])
                (loss / 3).backward()
                del loss
            observations.append(dict(role=item['role'], **metrics))
        pairs = template['pairs']
        error = template['items'][0]['row']
        for pair in pairs:
            guard()
            seed = operation_seed(train_seed, update, 'PREF', pair['edit_index'])
            with addressed_rng(T, model.device, seed):
                cpu_state = T.get_rng_state().clone()
                cuda_state = T.cuda.get_rng_state(model.device).clone()
                with T.autocast('cuda', dtype=T.bfloat16):
                    plus = model.content_score(error, features[error['case_id']], pair, True)
                    guard()
                    T.set_rng_state(cpu_state)
                    T.cuda.set_rng_state(cuda_state, model.device)
                    minus = model.content_score(error, features[error['case_id']], pair, False)
                    guard()
                    gap = plus - minus
                    preference = T.nn.functional.softplus(-gap)
                need(bool(T.isfinite(plus)) and bool(T.isfinite(minus)) and
                     bool(T.isfinite(preference)), 'NONFINITE_PREF_SCORE')
                # Deliberately no lambda==0 fast path or detached score.
                ((weight / len(pairs)) * preference).backward()
                gaps.append({'edit_index': pair['edit_index'],
                             'gap': float(gap.detach()), 'loss': float(preference.detach())})
                del plus, minus, gap, preference, cpu_state, cuda_state
            guard()
        stats, params = _diagnose_gradients(model, T)
        for group in ('LoRA', 'DenseBridge'):
            need(stats[group]['trainable_tensors'] > 0 and stats[group]['grad_none_tensors'] == 0
                 and stats[group]['grad_nonzero_tensors'] > 0, 'TRAINABLE_GROUP_GRADIENTS')
        norm = T.nn.utils.clip_grad_norm_(params, 1.0)
        need(bool(T.isfinite(norm)), 'NONFINITE_GLOBAL_GRADIENT')
        guard()
        progress['optimizer_update_inflight'] = update
        optimizer.step()
        progress['optimizer_updates'] = update
        progress['optimizer_update_inflight'] = None
        model.assert_frozen()
        T.cuda.synchronize(model.device)
        after = model.export()
        need(set(before) == set(after), 'ADAPTIVE_KEYS_CHANGED')
        changes, deltas = {'LoRA': 0, 'DenseBridge': 0}, {'LoRA': 0.0, 'DenseBridge': 0.0}
        for name, value in after.items():
            group = 'LoRA' if name.startswith('llm.') else 'DenseBridge'
            need(bool(T.isfinite(value).all()), 'NONFINITE_PARAMETER')
            changes[group] += int(not T.equal(before[name], value))
            delta = float((value.float() - before[name].float()).square().sum())
            need(math.isfinite(delta), 'NONFINITE_PARAMETER_DELTA')
            deltas[group] += delta
        need(all(n > 0 for n in changes.values()), 'TRAINABLE_GROUP_UNCHANGED')
        guard()
        return {'update': update, 'pool_cursor': template['pool_cursor'],
                'optimizer_updates_committed': 1, 'SFT': observations,
                'mean3_loss': sum(x['loss'] for x in observations) / 3,
                'eligible_edits': len(pairs), 'preference': gaps,
                'edit_status_counts': {status: sum(x['status'] == status for x in template['edit_statuses'])
                                       for status in sorted({x['status'] for x in template['edit_statuses']})},
                'lambda_pref': weight, 'preclip_gradient_norm': float(norm),
                'gradient_diagnostics': stats, 'changed_tensors': changes,
                'parameter_delta_l2': {k: math.sqrt(v) for k, v in deltas.items()}}
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise  # Entire once fails; a possibly committed optimizer step is never retried.
