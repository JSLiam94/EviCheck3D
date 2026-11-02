"""Retain all 864 regional CT tokens instead of averaging them to 64 slots."""
import torch
from torch import nn
import torch.nn.functional as F
from settings import O
from models import LocalBridge,Repair

class DenseBridge(LocalBridge):
    def __init__(self,width):
        super().__init__(width);self.position=nn.Parameter(torch.zeros(865,width))
    def forward(self,memory):
        assert memory.shape[1:]==(865,512)
        x=F.layer_norm(memory.float(),(512,));z=x@self.fixed+self.residual(x)
        return F.layer_norm(z,(z.shape[-1],))*self.log_scale.exp().clamp(max=.3)+self.position

class DenseRepair(Repair):
    def __init__(self,kind='spatial_dense',local_rank=0):
        super().__init__('spatial',local_rank)
        ck=torch.load(O/'spatial/epoch_2.pt',map_location='cpu',weights_only=True)
        state=ck['adaptive_state'];position=state['bridge.position'];width=position.shape[-1]
        local=F.interpolate(position[:64].reshape(4,4,4,width).permute(3,0,1,2)[None],size=(6,12,12),mode='trilinear',align_corners=False)[0]
        state['bridge.position']=torch.cat([local.permute(1,2,3,0).reshape(864,width),position[64:].mean(0,keepdim=True)],0)
        self.bridge=DenseBridge(width).to(self.device);self.n_prefix=865
        self.adaptive_names={n for n,p in self.named_parameters() if p.requires_grad};self.restore(state)
