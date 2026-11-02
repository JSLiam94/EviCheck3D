"""Pure helpers for the predeclared first-decision objective intervention."""
from __future__ import annotations

from typing import Any, Sequence


PROTOCOL = "center-first-decision-objective-v2"
DECISION_WEIGHT = 1.0
KEEP_PROGRAM = "[]"
ERROR_REPLACEMENT_WEIGHT = 0.8
ERROR_CONTEXT_WEIGHT = 0.2


def _token_sequence(value: Sequence[int], code: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(code)
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(code)
    return tuple(value)


def derive_shared_action_index(
    error_targets: Sequence[Sequence[int]], keep_target: Sequence[int]
) -> dict[str, Any]:
    """Find a shared EDIT-vs-KEEP divergence using TRAIN error targets only.

    Sequences must already include EOS. The intervention is refused unless all
    error targets first diverge from the KEEP target at one index, share the
    same preceding prefix, and begin the divergent branch with one token ID.
    """
    keep = _token_sequence(keep_target, "KEEP_TARGET_TOKENS_REFUSED")
    if not isinstance(error_targets, (list, tuple)) or not error_targets:
        raise ValueError("TRAIN_ERROR_TARGETS_REFUSED")

    action_indices: list[int] = []
    prefixes: list[tuple[int, ...]] = []
    branch_tokens: list[int] = []
    for raw in error_targets:
        target = _token_sequence(raw, "TRAIN_ERROR_TARGET_TOKENS_REFUSED")
        limit = min(len(target), len(keep))
        index = next((i for i in range(limit) if target[i] != keep[i]), None)
        if index is None:
            if len(target) == len(keep):
                raise ValueError("TRAIN_ERROR_DOES_NOT_DIVERGE_FROM_KEEP")
            index = limit
        if index >= len(target) or index >= len(keep):
            raise ValueError("ACTION_INDEX_OUT_OF_RANGE")
        action_indices.append(index)
        prefixes.append(target[:index])
        branch_tokens.append(target[index])

    if len(set(action_indices)) != 1:
        raise ValueError("TRAIN_ERROR_BRANCH_INDEX_MISMATCH")
    if len(set(prefixes)) != 1 or prefixes[0] != keep[:action_indices[0]]:
        raise ValueError("TRAIN_ERROR_BRANCH_PREFIX_MISMATCH")
    if len(set(branch_tokens)) != 1:
        raise ValueError("TRAIN_ERROR_BRANCH_TOKEN_FAMILY_MISMATCH")

    index = action_indices[0]
    return {
        "protocol": PROTOCOL,
        "action_index": index,
        "shared_prefix_token_ids": list(prefixes[0]),
        "edit_branch_token_id": branch_tokens[0],
        "keep_branch_token_id": keep[index],
        "keep_target_length": len(keep),
        "error_target_count": len(error_targets),
        "source_split": "TRAIN_FIT_ERROR_TARGETS_ONLY",
    }


def combine_objective_components(
    primary_type: int,
    *,
    mean_ce: Any,
    replacement_ce: Any | None,
    context_ce: Any | None,
    decision_ce: Any,
) -> dict[str, Any]:
    """Combine the frozen V1.1 sequence loss with fixed-weight decision CE."""
    if type(primary_type) is not int or primary_type < 0:
        raise ValueError("PRIMARY_TYPE_REFUSED")
    if decision_ce is None:
        raise ValueError("DECISION_CE_REQUIRED")
    if primary_type == 0:
        if mean_ce is None:
            raise ValueError("NORMAL_MEAN_CE_REQUIRED")
        original_loss = mean_ce
    else:
        if replacement_ce is None or context_ce is None:
            raise ValueError("ERROR_CE_COMPONENTS_REQUIRED")
        original_loss = (
            ERROR_REPLACEMENT_WEIGHT * replacement_ce
            + ERROR_CONTEXT_WEIGHT * context_ce
        )
    return {
        "original_loss": original_loss,
        "decision_ce": decision_ce,
        "decision_weight": DECISION_WEIGHT,
        "total_loss": original_loss + DECISION_WEIGHT * decision_ce,
        "mean_ce": mean_ce,
        "replacement_ce": replacement_ce,
        "context_ce": context_ce,
    }
