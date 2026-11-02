"""Online frozen CT-CLIP only, CPU featureLRU16, no whole-pool cache writes."""
from collections import OrderedDict
import random
from pathlib import Path
from pref_inputs import need


class OnlineFrozenFeatures:
    def __init__(self,T,device,pool,guard,encoder_factory,max_features=16):
        need(max_features==16,'FEATURE_LRU16')
        self.T=T;self.device=device;self.pool=pool;self.guard=guard;self.cache=OrderedDict();self.max_features=max_features
        # Use the same source-pinned pretrained CT-CLIP/preprocessing factory as
        # the committed feature provenance. Never import DirectFullRepair or a
        # repair-trained vision checkpoint. Root authenticates this factory.
        self.encoder=encoder_factory('cpu');self.encoder.requires_grad_(False);self.encoder.eval();self.encoded_CT=0;self.cache_hits=0
        self.encoder.to('cpu');self.encoder.device=T.device('cpu')
        need(all(not p.requires_grad for p in self.encoder.parameters()),'FROZEN_CT_ENCODER_ONLY')

    def __getitem__(self,case_id):
        self.guard();row=self.pool.by_id(case_id);key=row['canonical_ct_id']
        if key in self.cache:
            self.cache.move_to_end(key);self.cache_hits+=1;return self.cache[key]
        T=self.T;need(Path(row['image_memory']).is_absolute(),'FULL_TRAIN_CT_PATH')
        self.encoder.to(self.device);self.encoder.device=self.device;self.encoder.eval()
        versions={n:p._version for n,p in self.encoder.named_parameters()}
        try:
            python_rng=random.getstate()
            try:
                with T.random.fork_rng(devices=[self.device.index],enabled=True),T.no_grad(),T.autocast('cuda',dtype=T.bfloat16):x=self.encoder(row['image_memory']).detach().float().cpu().contiguous()
            finally:random.setstate(python_rng)
            need(tuple(x.shape)==(865,512) and x.dtype==T.float32 and x.device.type=='cpu' and not x.requires_grad and bool(T.isfinite(x).all()),'FULL_TRAIN_ONLINE_FEATURE')
            need(all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in self.encoder.named_parameters()),'FROZEN_ONLINE_CT_CHANGED')
        finally:
            self.encoder.to('cpu');self.encoder.device=T.device('cpu');T.cuda.empty_cache()
        self.encoded_CT+=1;self.cache[key]=x
        while len(self.cache)>self.max_features:self.cache.popitem(last=False)
        self.guard();return x

    def prepare(self,template):
        # Encode before the LM gradient graph is created. CLEAN reuses E CT.
        for item in template['items']:self[item['row']['case_id']]

    def safe(self):
        return dict(online_CT_encodes=self.encoded_CT,feature_cache_hits=self.cache_hits,feature_cache_capacity=self.max_features,
          feature_files_written=0,feature_shape=[865,512],feature_dtype='float32',CT_encoder_frozen=True,LM_hidden_cached=False,bridge_cached=False)
