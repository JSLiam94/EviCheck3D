"""Adapter for the official CT-CLIP visual encoder/checkpoint."""

from __future__ import annotations

import os
import sys
import importlib
import contextlib
import io
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLIPVisualAdapter(nn.Module):
    """Use CT-CLIP visual features when the official code/checkpoint are available.

    The official CT-CLIP repository is not packaged as a normal Transformers model.
    This adapter imports the cloned code from `pretrained/CT-CLIP`, instantiates the
    published CTViT+CTCLIP architecture, and exposes a dense feature map compatible
    with the rest of this project. Initialization failure aborts the public entry points; no fallback CNN is used.
    """

    def __init__(
        self,
        project_root: str,
        hidden_dim: int = 512,
        code_dir: str = "pretrained/CT-CLIP",
        checkpoint_path: str = "pretrained/ct_clip/models/CT-CLIP-Related/CT-CLIP_v2.pt",
        text_model_path: str = "pretrained/ct_clip_text",
        input_shape_zyx: list[int] | tuple[int, int, int] = (240, 480, 480),
        freeze: bool = True,
        train_visual_encoder: bool = False,
        train_visual_projection: bool = True,
        allow_fallback: bool = False,
        latent_dim: int = 512,
        suppress_output: bool = True,
        autocast: bool = True,
        autocast_dtype: str = "bfloat16",
        checkpoint_min_loaded_ratio: float = 0.20,
        checkpoint_require_visual: bool = True,
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.hidden_dim = int(hidden_dim)
        self.code_dir = self._resolve(code_dir)
        self.checkpoint_path = self._resolve(checkpoint_path)
        self.text_model_path = self._resolve(text_model_path)
        self.input_shape_zyx = tuple(int(x) for x in input_shape_zyx)
        self.freeze = bool(freeze)
        self.train_visual_encoder = bool(train_visual_encoder)
        self.train_visual_projection = bool(train_visual_projection)
        self.allow_fallback = bool(allow_fallback)
        self.latent_dim = int(latent_dim)
        self.suppress_output = bool(suppress_output)
        self.autocast = bool(autocast)
        self.autocast_dtype = str(autocast_dtype)
        self.checkpoint_min_loaded_ratio = float(checkpoint_min_loaded_ratio)
        self.checkpoint_require_visual = bool(checkpoint_require_visual)
        self.backend_name = "ct_clip"
        self.init_error = ""
        self.checkpoint_load_info: Dict[str, object] = {
            "loaded": False,
            "path": self.checkpoint_path,
        }

        self.model = self._build_official_model()
        if self.model is None:
            if not self.allow_fallback:
                raise RuntimeError(
                    "CT-CLIP could not be initialized. Provide the official local model assets "
                    "and install CT-CLIP optional dependencies, or set allow_fallback=true. "
                    f"Original error: {self.init_error}"
                )
            self.backend_name = "tiny_fallback"
            raise RuntimeError('Fallback CNN is not included in this source distribution')
        else:
            self.proj = nn.Linear(self.latent_dim, self.hidden_dim)
            self._configure_trainable_official_modules()

    def _set_trainable(self, module: nn.Module | None, trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def _configure_trainable_official_modules(self) -> None:
        if self.model is None:
            return
        if self.freeze:
            self._set_trainable(self.model, False)
            if self.train_visual_encoder:
                self._set_trainable(getattr(self.model, "visual_transformer", None), True)
                if self.train_visual_projection:
                    self._set_trainable(getattr(self.model, "to_visual_latent", None), True)
                    self._set_trainable(getattr(self.model, "to_visual_latent_extra", None), True)
        elif not self.train_visual_projection:
            self._set_trainable(getattr(self.model, "to_visual_latent", None), False)
            self._set_trainable(getattr(self.model, "to_visual_latent_extra", None), False)

    @property
    def _official_visual_encoder_has_trainable_params(self) -> bool:
        if self.model is None:
            return False
        visual = getattr(self.model, "visual_transformer", None)
        if visual is None:
            return False
        return any(param.requires_grad for param in visual.parameters())

    def _resolve(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self.project_root) / p)

    def _add_official_paths(self) -> None:
        for sub in ["CT_CLIP", "transformer_maskgit"]:
            p = str(Path(self.code_dir) / sub)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

    def _build_official_model(self):
        try:
            self._add_official_paths()
            from ct_clip import CTCLIP
            from transformer_maskgit import CTViT
            from transformers import BertModel, BertTokenizer

            local_text = os.path.isdir(self.text_model_path)
            if not local_text:
                raise FileNotFoundError(
                    f"CT-CLIP text model directory is missing: {self.text_model_path}. "
                    "Provide the official local model assets before training."
                )
            text_source = self.text_model_path
            tokenizer = BertTokenizer.from_pretrained(
                text_source,
                do_lower_case=True,
                local_files_only=True,
                trust_remote_code=True,
            )
            text_encoder = BertModel.from_pretrained(
                text_source,
                local_files_only=True,
                trust_remote_code=True,
            )
            text_encoder.resize_token_embeddings(len(tokenizer))
            image_encoder = CTViT(
                dim=512,
                codebook_size=8192,
                image_size=480,
                patch_size=20,
                temporal_patch_size=10,
                spatial_depth=4,
                temporal_depth=4,
                dim_head=32,
                heads=8,
            )
            ct_clip_module = importlib.import_module("ct_clip.ct_clip")
            original_from_pretrained = ct_clip_module.BertTokenizer.from_pretrained

            def _local_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                if str(pretrained_model_name_or_path) == "microsoft/BiomedVLP-CXR-BERT-specialized":
                    pretrained_model_name_or_path = text_source
                    kwargs["local_files_only"] = True
                    kwargs["trust_remote_code"] = True
                return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

            ct_clip_module.BertTokenizer.from_pretrained = classmethod(_local_from_pretrained)
            try:
                clip = CTCLIP(
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    dim_image=294912,
                    dim_text=768,
                    dim_latent=self.latent_dim,
                    extra_latent_projection=False,
                    use_mlm=False,
                    downsample_image_embeds=False,
                    use_all_token_embeds=False,
                )
            finally:
                ct_clip_module.BertTokenizer.from_pretrained = original_from_pretrained
            if os.path.exists(self.checkpoint_path):
                self._load_checkpoint_flexible(clip, self.checkpoint_path)
            else:
                raise FileNotFoundError(self.checkpoint_path)
            return clip
        except Exception as exc:
            self.init_error = repr(exc)
            print(f"[WARN] CT-CLIP backend unavailable, falling back if allowed: {exc}", flush=True)
            return None

    @staticmethod
    def _candidate_checkpoint_keys(key: str) -> list[str]:
        k = str(key)
        candidates = [k]
        prefixes = (
            "module.",
            "model.",
            "clip.",
            "ct_clip.",
            "CTClip.",
            "CTClip.module.",
            "image_encoder.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix) :]
                    candidates.append(k)
                    changed = True
        return candidates

    def _load_checkpoint_flexible(self, model: nn.Module, checkpoint_path: str) -> None:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        else:
            state = ckpt
        model_state = model.state_dict()
        cleaned = {}
        skipped_shape = []
        raw_keys = 0
        for key, value in state.items():
            raw_keys += 1
            if str(key).endswith("position_ids"):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            matched_key = None
            for candidate in self._candidate_checkpoint_keys(str(key)):
                if candidate in model_state:
                    matched_key = candidate
                    break
            if matched_key is None:
                continue
            if tuple(model_state[matched_key].shape) != tuple(value.shape):
                skipped_shape.append(
                    {
                        "key": matched_key,
                        "checkpoint_shape": tuple(value.shape),
                        "model_shape": tuple(model_state[matched_key].shape),
                    }
                )
                continue
            cleaned[matched_key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        unexpected = [k for k in unexpected if not str(k).endswith("position_ids")]
        loaded_keys = set(cleaned.keys())
        model_keys = {
            key for key in model_state.keys() if not str(key).endswith("position_ids")
        }
        loaded_ratio = len(loaded_keys) / max(len(model_keys), 1)
        visual_loaded = sum(
            1 for key in loaded_keys if str(key).startswith("visual_transformer.")
        )
        visual_total = sum(
            1 for key in model_keys if str(key).startswith("visual_transformer.")
        )
        projection_loaded = sum(
            1
            for key in loaded_keys
            if str(key).startswith("to_visual_latent")
            or str(key).startswith("to_visual_latent_extra")
        )
        self.checkpoint_load_info = {
            "loaded": True,
            "path": checkpoint_path,
            "raw_checkpoint_keys": int(raw_keys),
            "matched_keys": int(len(loaded_keys)),
            "model_keys": int(len(model_keys)),
            "matched_ratio": float(loaded_ratio),
            "visual_matched_keys": int(visual_loaded),
            "visual_total_keys": int(visual_total),
            "visual_projection_matched_keys": int(projection_loaded),
            "missing_keys": int(len(missing)),
            "unexpected_keys": int(len(unexpected)),
            "shape_mismatch_keys": int(len(skipped_shape)),
            "first_missing": list(missing[:20]),
            "first_unexpected": list(unexpected[:20]),
            "first_shape_mismatch": skipped_shape[:10],
        }
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Loaded CT-CLIP checkpoint: {self.checkpoint_load_info}", flush=True)
        if loaded_ratio < self.checkpoint_min_loaded_ratio:
            raise RuntimeError(
                "CT-CLIP checkpoint matched too few model keys: "
                f"{len(loaded_keys)}/{len(model_keys)} "
                f"({loaded_ratio:.3f}) from {checkpoint_path}. "
                "This usually means the checkpoint is not the official CT-CLIP "
                "state dict or the architecture does not match."
            )
        if self.checkpoint_require_visual and visual_loaded <= 0:
            raise RuntimeError(
                "CT-CLIP checkpoint did not load any visual_transformer keys "
                f"from {checkpoint_path}; refusing to train with random CT encoder."
            )

    def _resize_for_ct_clip(self, ct: torch.Tensor) -> torch.Tensor:
        # Project data layout from (B,1,Z,Y,X) to CT-CLIP's expected spatial volume.
        if tuple(ct.shape[-3:]) == self.input_shape_zyx:
            return ct
        return F.interpolate(ct, size=self.input_shape_zyx, mode="trilinear", align_corners=False)

    def _autocast_context(self, device: torch.device):
        enabled = self.autocast and device.type == "cuda"
        dtype = torch.bfloat16 if self.autocast_dtype == "bfloat16" else torch.float16
        try:
            return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)

    def forward(self, ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.backend_name == "tiny_fallback":
            return self.model(ct)

        official_visual_trainable = self._official_visual_encoder_has_trainable_params
        was_training = self.model.training
        if not official_visual_trainable:
            self.model.eval()
        image = self._resize_for_ct_clip(ct)
        device = image.device
        text_tokens = self.model.tokenizer(
            ["abdomen CT report"] * image.shape[0],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(device)
        context = torch.enable_grad() if official_visual_trainable else torch.no_grad()
        with context:
            with self._autocast_context(device):
                if self.suppress_output:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, image_latents, encoded_tokens = self.model(text_tokens, image, device=device, return_latents=True)
                else:
                    _, image_latents, encoded_tokens = self.model(text_tokens, image, device=device, return_latents=True)
        if was_training and official_visual_trainable:
            self.model.train()

        if isinstance(encoded_tokens, torch.Tensor) and encoded_tokens.ndim == 5:
            # Official CT-CLIP returns (B,H,W,Z,C) tokens in current code path.
            feat = encoded_tokens.permute(0, 4, 3, 1, 2).contiguous()
            if feat.shape[1] != self.hidden_dim:
                b, c, z, y, x = feat.shape
                feat = feat.to(dtype=self.proj.weight.dtype)
                feat = self.proj(feat.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
            else:
                feat = feat.to(dtype=self.proj.weight.dtype)
            image_latents = image_latents.to(dtype=self.proj.weight.dtype)
            return {"dense_feat": feat, "aux": {"ct_clip_latents": image_latents}}

        image_latents = image_latents.to(dtype=self.proj.weight.dtype)
        feat = self.proj(image_latents).view(image.shape[0], self.hidden_dim, 1, 1, 1)
        return {"dense_feat": feat, "aux": {"ct_clip_latents": image_latents}}
