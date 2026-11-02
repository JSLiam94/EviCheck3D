"""Source-only adapter: three separate forwards/backwards per optimizer update.
No torch/model/tokenizer/data import, optimizer call, file write or launch API.
Caller retains accepted zero_grad, finite-gradient, clipping, frozen-parameter,
optimizer-step, checkpoint/RNG/cursor, resource and hardware-lease boundaries.
"""
from __future__ import annotations


def tuple_backward(model, items, *, action_index, feature_lookup, autocast, guard):
    """Accumulate mean of three unchanged V2 losses without retaining graphs.

    Call only after caller zero_grad. Caller must abort the whole update on any
    exception before optimizer.step. Feature lookup is keyed by the compiled
    host-private plan, never by target text. No telemetry contains row strings.
    """
    if type(items) not in (list, tuple) or len(items) != 3:
        raise ValueError('THREE_EXPOSURES_REQUIRED')
    if type(action_index) is not int or action_index < 0:
        raise ValueError('ACTION_INDEX_REFUSED')
    observations = []
    for item in items:
        guard()
        row = item['row']
        if row.get('internal_split') != 'fit' or row.get('target_status') != 'READY':
            raise ValueError('READY_TRAIN_FIT_ONLY')
        features = feature_lookup(item['feature_case_id'])
        with autocast():
            loss, ce, mask, components = model.losses(row, features, action_index=action_index)
        # Derive finite check without importing a new numerical/runtime stack.
        values = [loss.detach(), ce.detach().mean(), components['decision_ce'].detach()]
        if not all(bool(value.isfinite().all()) for value in values):
            raise ValueError('NONFINITE_TUPLE_LOSS')
        (loss / 3.0).backward()
        observations.append(dict(role=item['role'], loss=float(values[0]),
                                 target_NLL=float(values[1]), decision_CE=float(values[2]),
                                 target_tokens=len(ce), replacement_tokens=int(mask.sum())))
        del loss, ce, mask, components, values, features
        guard()
    return dict(loss=sum(item['loss'] for item in observations) / 3.0,
                exposures=3, per_exposure=observations,
                optimizer_updates_committed=0)
