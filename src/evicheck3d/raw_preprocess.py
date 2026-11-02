"""Canonical CT crop; fill the XY field while resampling the slice axis separately."""
import numpy as np
import nibabel as nib
from scipy import ndimage
import torch
import torch.nn.functional as F

def volume(path,shape=(64,256,256),window=(-100.,400.)):
    image=nib.as_closest_canonical(nib.load(path));spacing=np.array(image.header.get_zooms()[:3])
    x=np.nan_to_num(image.get_fdata(dtype=np.float32),nan=-1000,posinf=2000,neginf=-1000)
    coarse=x[::4,::4,::4]>-500
    components,n=ndimage.label(coarse)
    if n:
        counts=np.bincount(components.ravel());counts[0]=0;where=np.where(components==counts.argmax())
        slices=[]
        for axis,idx in enumerate(where):
            margin=int(np.ceil(30/spacing[axis]));start=max(0,int(idx.min())*4-margin);end=min(x.shape[axis],(int(idx.max())+1)*4+margin)
            slices.append(slice(start,end))
        x=x[tuple(slices)]
    x=np.ascontiguousarray(x.transpose(2,1,0));physical=np.array(x.shape)*spacing[::-1]
    d,h,w=shape;scale=min(h/physical[1],w/physical[2])
    target=(d,max(1,min(h,round(physical[1]*scale))),max(1,min(w,round(physical[2]*scale))))
    low,high=window;x=np.clip((x-low)/(high-low),0,1)
    with torch.inference_mode():
        y=F.interpolate(torch.from_numpy(x)[None,None],size=target,mode='trilinear',align_corners=False)[0]
        py=h-target[1];px=w-target[2];y=F.pad(y,(px//2,px-px//2,py//2,py-py//2),value=0)
    audit=dict(nonpadding_xy_fraction=target[1]*target[2]/(h*w),
        old_isotropic_letterbox_xy_fraction=float(np.prod(np.minimum(np.array(shape),np.maximum(1,np.round(physical*min(np.array(shape)/physical))))[1:])/h/w),
        canonical_orientation='RAS',shape=list(shape),finite=bool(y.isfinite().all()))
    assert y.shape==(1,*shape) and y.isfinite().all()
    return y,audit
