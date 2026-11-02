from pathlib import Path
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from raw_preprocess import volume
class CTEncoder(nn.Module):
    def __init__(self,device):
        super().__init__();self.device=device
        from ct_clip_adapter import CTCLIPVisualAdapter
        adapter=CTCLIPVisualAdapter(str(Path.cwd()),hidden_dim=512,code_dir=os.environ['EVICHECK3D_CT_CLIP_CODE'],checkpoint_path=os.environ['EVICHECK3D_CT_CLIP_CHECKPOINT'],text_model_path=os.environ['EVICHECK3D_CT_CLIP_TEXT'],allow_fallback=False)
        # The quantizer disables autocast and expects its codebook in FP32.
        # Preserve pretrained precision; matrix operations still use BF16 AMP.
        adapter.requires_grad_(False);adapter.to(device=device)
        self.tower=adapter.model.visual_transformer;self.latent=adapter.model.to_visual_latent;del adapter
        self.tower.requires_grad_(False);self.latent.requires_grad_(False)
        for name in ('enc_spatial_transformer','enc_temporal_transformer'):
            layers=getattr(self.tower,name).layers
            for layer in layers[-2:]:layer.requires_grad_(True);layer.float()
        self.frozen_versions={n:p._version for n,p in self.named_parameters() if not p.requires_grad}
        self.train(True)
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'tower'):
            self.tower.eval();self.latent.eval()
            for name in ('enc_spatial_transformer','enc_temporal_transformer'):
                for layer in getattr(self.tower,name).layers[-2:]:layer.train(mode)
            self.tower.vq.eval()
        return self
    def forward(self,path):
        raw,_=volume(path,shape=(240,480,480));x=(2*raw.clone()-1)[None].to(self.device)
        tokens=self.tower.to_patch_emb(x)
        pre=checkpoint(self.tower.encode,tokens,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.tower.encode(tokens)
        # Keep the original quantizer/codebook fixed. Its nearest-code lookup
        # alone has no input gradient in eval mode; use an explicit STE so
        # repair supervision reaches the trained CT encoder blocks.
        with torch.no_grad():quantized,_,_=self.tower.vq(pre.reshape(len(pre),-1,512))
        encoded=pre+(quantized.reshape_as(pre)-pre).detach()
        local=F.adaptive_avg_pool3d(encoded.permute(0,4,1,2,3).float(),(6,12,12)).flatten(2).transpose(1,2)[0]
        global_token=F.normalize(self.latent(encoded.mean(1).reshape(1,-1)).float(),dim=-1)[0]
        return torch.cat([local,global_token[None]],0)
    def assert_frozen(self):
        for n,p in self.named_parameters():
            if n in self.frozen_versions:assert p._version==self.frozen_versions[n] and not p.requires_grad and p.grad is None
