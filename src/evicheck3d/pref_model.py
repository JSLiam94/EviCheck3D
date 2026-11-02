"""Admitted-only model delta. This file is never imported by source review.
Inherited model/architecture files remain byte-identical. Shared SFT deletion
boundary and complete padded candidate forwards are the only model deltas.
"""
from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from main96_model import Main96UnitProgramRepair
from pref_inputs import need


class ContentPreferenceRepair(Main96UnitProgramRepair):
    def __init__(self):
        super().__init__('bound_ct')
        # Make the checkpoint recomputation policy explicit for every LLM block.
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True})

    def _target_ce(self, report, features, target, expected_ids, *, common_length=None,
                   prompt_length=None, force_offload=False):
        """Causal prediction at prompt_len-1+i, including target EOS; never pad
        by duplicating EOS attention. Both branches have equal matrix shapes.
        Offload encloses pack too, including trainable DenseBridge activations.
        """
        length = self.prompt_token_count(report) + len(expected_ids) + 1
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if force_offload or length > 8192 else contextlib.nullcontext())
        with offload:
            inputs, _, start, wanted = self.pack(report, features, target)
            need(wanted[:-1] == expected_ids and wanted[-1] == self.tokenizer.eos_token_id,
                 'EXPECTED_TARGET_EOS_ALIGNMENT')
            if prompt_length is not None:
                need(start == prompt_length, 'PROMPT_LENGTH_BINDING')
            real_length = inputs['inputs_embeds'].shape[1]
            need(real_length == start + len(wanted), 'REAL_MATRIX_LENGTH')
            size = real_length if common_length is None else common_length
            need(real_length <= size <= 32768, 'PADDED_CONTEXT_LENGTH')
            extra = size - real_length
            if extra:
                embed = inputs['inputs_embeds']
                inputs['inputs_embeds'] = torch.cat(
                    [embed, embed.new_zeros((1, extra, embed.shape[-1]))], dim=1)
                mask = inputs['attention_mask']
                inputs['attention_mask'] = torch.cat([mask, mask.new_zeros((1, extra))], dim=1)
            # Real tokens retain 0..real_length-1. Masked right pads have unused 0.
            positions = torch.arange(size, device=self.device).unsqueeze(0)
            inputs['position_ids'] = positions.masked_fill(inputs['attention_mask'] == 0, 0)
            base = self.llm.get_base_model()
            hidden_all = base.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            hidden = hidden_all[0, start - 1:start + len(wanted) - 1]
            need(len(hidden) == len(wanted), 'CAUSAL_TARGET_ALIGNMENT')
            labels = torch.tensor(wanted, device=self.device)
            pieces = []
            for offset in range(0, len(wanted), 128):
                def token_ce(h, y):
                    return F.cross_entropy(base.lm_head(h).float(), y, reduction='none')
                pieces.append(checkpoint(token_ce, hidden[offset:offset + 128],
                                         labels[offset:offset + 128], use_reentrant=False,
                                         preserve_rng_state=True))
            return torch.cat(pieces)

    def sft_loss(self, row, features, layout, action_index):
        ce = self._target_ce(row['report'], features, row['target'], layout['ids'],
                             prompt_length=layout['prompt_length'])
        need(0 <= action_index < len(ce), 'ACTION_INDEX_ALIGNMENT')
        mask = torch.tensor(layout['replacement_mask'] + [False], device=self.device,
                            dtype=torch.bool)
        if row['primary_type'] == 0 or layout['pure_deletion']:
            original = ce.mean()
        else:
            need(bool(mask.any()) and bool((~mask).any()), 'SFT_TOKEN_MASK')
            original = 0.8 * ce[mask].mean() + 0.2 * ce[~mask].mean()
        loss = original + ce[action_index]
        need(bool(torch.isfinite(ce).all()) and bool(torch.isfinite(loss)), 'NONFINITE_SFT')
        return loss, {'loss': float(loss.detach()), 'target_NLL': float(ce.detach().mean()),
                      'action_CE': float(ce[action_index].detach()),
                      'target_tokens_including_EOS': len(ce),
                      'overlap_mask_tokens': int(mask.sum()),
                      'pure_deletion_common_fallback': bool(layout['pure_deletion'])}

    def content_score(self, row, features, pair, positive):
        prefix = 'positive' if positive else 'negative'
        ids = pair[prefix + '_ids']
        indices = pair[prefix + '_indices']
        need(indices and all(type(i) is int and 0 <= i < len(ids) for i in indices),
             'CONTENT_INDICES_EXCLUDE_EOS_PAD')
        ce = self._target_ce(row['report'], features, pair[prefix + '_target'], ids,
                             common_length=pair['common_length'],
                             prompt_length=pair['prompt_length'], force_offload=True)
        need(bool(torch.isfinite(ce).all()), 'NONFINITE_PREF_TARGET_CE')
        return -ce[torch.tensor(indices, device=self.device)].mean()
