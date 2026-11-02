"""Two supervised repair foundations sharing one exact patch execution contract."""
import itertools,json,math,sys,os
import torch
from torch import nn
import torch.nn.functional as F
from settings import R,C,O,RAD,OLD
from edit_codec import apply_edits
from weighted_edit_codec import serialize_with_replacement_ranges

class LocalBridge(nn.Module):
    def __init__(self,width):
        super().__init__()
        generator=torch.Generator().manual_seed(20261021)
        # A fixed, bias-free residual keeps case differences present before
        # supervision learns to use them. No learned-query average bottleneck.
        self.register_buffer('fixed',torch.randn(512,width,generator=generator)/math.sqrt(512))
        self.residual=nn.Sequential(nn.Linear(512,512,bias=False),nn.GELU(),nn.Linear(512,width,bias=False))
        nn.init.zeros_(self.residual[-1].weight)
        self.position=nn.Parameter(torch.randn(68,width)*.001)
        self.log_scale=nn.Parameter(torch.tensor(math.log(.04)))
    def forward(self,memory):
        b=len(memory);spatial=memory[:,:864].float().transpose(1,2).reshape(b,512,6,12,12)
        local=F.adaptive_avg_pool3d(spatial,(4,4,4)).flatten(2).transpose(1,2)
        global_token=memory[:,-1:].float().repeat(1,4,1)
        x=F.layer_norm(torch.cat([local,global_token],1),(512,))
        z=x@self.fixed+self.residual(x)
        prefix=F.layer_norm(z,(z.shape[-1],))*self.log_scale.exp().clamp(max=.3)+self.position
        return prefix

class Repair(nn.Module):
    def __init__(self,kind,local_rank=0):
        super().__init__();self.kind=kind;self.device=torch.device('cuda',local_rank);torch.cuda.set_device(self.device)
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        if kind=='spatial':
            base=os.environ['EVICHECK3D_BASE_MODEL']
            self.tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
            if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
            q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
            llm=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,quantization_config=q,device_map={'':local_rank},dtype=torch.bfloat16,attn_implementation='sdpa')
            llm=prepare_model_for_kbit_training(llm,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
            self.bridge=LocalBridge(llm.config.hidden_size).to(self.device);self.n_prefix=68
        else:
            sys.path[:0]=[str(C/'deps'),str(RAD/'src')]
            from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM,load_radfm_checkpoint
            self.tokenizer=AutoTokenizer.from_pretrained(RAD/'Quick_demo/Language_files',local_files_only=True,use_fast=False,unk_token='<unk>',bos_token='<s>',eos_token='</s>')
            assert (self.tokenizer.vocab_size,self.tokenizer.unk_token_id,self.tokenizer.bos_token_id,self.tokenizer.eos_token_id)==(32000,0,1,2)
            self.tokenizer.add_special_tokens({'additional_special_tokens':['<image>','</image>',*[f'<image{i}>' for i in range(32)]]})
            assert self.tokenizer.convert_tokens_to_ids(['<image>','</image>','<image0>','<image31>'])==[32000,32001,32002,32033]
            self.tokenizer.pad_token_id=0
            native=MultiLLaMAForCausalLM(C/'native_language_config.json',RAD/'Quick_demo/MedKEBERT/config.json',pad_token_id=0,bos_token_id=1,eos_token_id=2)
            load_radfm_checkpoint(native,RAD/'models/pytorch_model.bin');native.requires_grad_(False)
            llm=native.lang_model.to(dtype=torch.bfloat16,device=self.device)
            self.register_buffer('figure_tokens',native.embedding_layer.figure_token_weight.detach().to(self.device,dtype=torch.bfloat16).clone())
            del native;torch.cuda.empty_cache()
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.bridge=nn.Sequential(nn.Linear(5120,256),nn.GELU(),nn.Linear(256,5120)).to(self.device)
            nn.init.zeros_(self.bridge[-1].weight);nn.init.zeros_(self.bridge[-1].bias);self.n_prefix=32
        llm.requires_grad_(False)
        self.llm=get_peft_model(llm,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        self.llm.config.use_cache=False;self.max_context=4096
        self.frozen_versions={n:p._version for n,p in self.llm.named_parameters() if not p.requires_grad}
        self.adaptive_names={n for n,p in self.named_parameters() if p.requires_grad};self.alignment_mode=False
        self.register_buffer('dummy',torch.zeros((),device=self.device),persistent=False)
    def set_alignment(self,enabled):
        self.alignment_mode=enabled
        for n,p in self.named_parameters():p.requires_grad_(n in self.adaptive_names and (not enabled or n.startswith('bridge.')))
    def prompt(self,report):
        if self.alignment_mode:
            if self.kind=='spatial':return self.tokenizer.apply_chat_template([dict(role='system',content='Describe this abdominal CT examination. Write only the radiology report.'),dict(role='user',content='CT evidence:\n<CT_EVIDENCE>\nWrite the report:')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return '<image>'+''.join(f'<image{i}>' for i in range(32))+'</image>\nDescribe this abdominal CT examination:\n'
        instruction='Correct factual errors in this abdominal CT report using the CT evidence. Return only a JSON array of minimal edits with exactly "old" and "new" strings. Each "old" must be a unique exact span of the original report. Anchor insertions with unchanged adjacent words. Do not overlap edits. Preserve supported findings and unrelated wording. If no edit is supported, return [].'
        if self.kind=='spatial':
            return self.tokenizer.apply_chat_template([dict(role='system',content=instruction),dict(role='user',content='CT evidence:\n<CT_EVIDENCE>\nReport:\n'+report)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return '<image>'+''.join(f'<image{i}>' for i in range(32))+'</image>\n'+instruction+'\nReport:\n'+report+'\nEdits:\n'
    def pack(self,report,memory,target=None):
        tok=self.tokenizer;text=self.prompt(report)
        if self.kind=='spatial':
            before,after=text.split('<CT_EVIDENCE>');a=tok(before,add_special_tokens=False)['input_ids'];b=tok(after,add_special_tokens=False)['input_ids']
            prompt=a+[tok.pad_token_id]*self.n_prefix+b
        else:prompt=tok(text,add_special_tokens=True)['input_ids']
        wanted=[] if target is None else tok(target,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
        assert len(prompt)+len(wanted)<=self.max_context,'No context truncation allowed'
        ids=torch.tensor([prompt+wanted],device=self.device)
        if self.kind=='spatial':
            embedding=self.llm.get_input_embeddings()(ids);visual=self.bridge(memory[None].to(self.device))
            embedding=torch.cat([embedding[:,:len(a)],visual.to(embedding.dtype),embedding[:,len(a)+self.n_prefix:]],1)
        else:
            visual=memory[None].to(self.device).float();visual=visual+.1*self.bridge(visual)
            embedding=self.llm.get_input_embeddings()(ids.clamp_max(31999))
            figures=self.figure_tokens[(ids-32000).clamp(0,1)]
            embedding=torch.where(((ids>=32000)&(ids<32002))[...,None],figures,embedding)
            selected=torch.gather(visual,1,(ids-32002).clamp(0,31)[...,None].expand(-1,-1,5120))
            embedding=torch.where((ids>=32002)[...,None],selected.to(embedding.dtype),embedding)
            assert int((ids>=32002).sum())==32
        return dict(inputs_embeds=embedding,attention_mask=torch.ones_like(ids)),ids,len(prompt),wanted
    def losses(self,row,memory=None):
        target,ranges=serialize_with_replacement_ranges(row['report'],row['reference'])
        inputs,_,_,ids=self.pack(row['report'],row['image_memory'] if memory is None else memory,target)
        logits=self.llm(**inputs,use_cache=False,logits_to_keep=len(ids)+1).logits[0,:-1].float()
        ce=F.cross_entropy(logits,torch.tensor(ids,device=self.device),reduction='none')
        offsets=self.tokenizer(target,add_special_tokens=False,return_offsets_mapping=True) if getattr(self.tokenizer,'is_fast',False) else None
        mask=torch.zeros(len(ids),device=self.device,dtype=torch.bool)
        if offsets:
            assert offsets['input_ids']==ids[:-1]
            for i,(a,b) in enumerate(offsets['offset_mapping']):
                if any(a<end and b>start for start,end in ranges):mask[i]=True
        else:
            # SentencePiece has no offsets: obtain exact token-prefix boundaries
            # conservatively, including a boundary token when segmentation joins.
            for a,b in ranges:
                start=max(0,len(self.tokenizer(target[:a],add_special_tokens=False)['input_ids'])-1)
                end=len(self.tokenizer(target[:b],add_special_tokens=False)['input_ids'])
                mask[start:end]=True
        keep=self.tokenizer('[]',add_special_tokens=False)['input_ids']+[self.tokenizer.eos_token_id]
        edit=self.tokenizer('[{"old":"X","new":"Y"}]',add_special_tokens=False)['input_ids']+[self.tokenizer.eos_token_id]
        decision=next(i for i,(a,b) in enumerate(itertools.zip_longest(keep,edit)) if a!=b)
        assert ids[decision]==(edit[decision] if ranges else keep[decision]),'Decision boundary is inconsistent'
        context=~mask;context[decision]=False
        loss=ce[decision]+(.75*ce[mask].mean() if mask.any() else 0)+(.15*ce[context].mean() if context.any() else 0)
        return loss,ce,mask
    def objective(self,row,wrong=None):
        if self.alignment_mode:
            inputs,_,_,ids=self.pack('',row['image_memory'],row['reference'])
            logits=self.llm(**inputs,use_cache=False,logits_to_keep=len(ids)+1).logits[0,:-1].float()
            loss=F.cross_entropy(logits,torch.tensor(ids,device=self.device));return loss,dict(token_ce=float(loss.detach()),ranking=0.)
        loss,ce,mask=self.losses(row);ranking=loss*0
        if wrong is not None and row['error'] and mask.any():
            _,negative,_=self.losses(row,wrong)
            ranking=F.relu(.1+ce[mask].mean()-negative[mask].mean());loss=loss+.2*ranking
        return loss,dict(token_ce=float(ce.detach().mean()),ranking=float(ranking.detach()))
    def trainable(self):return {n:p for n,p in self.named_parameters() if p.requires_grad}
    def export(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if n in self.adaptive_names}
    def restore(self,state):
        params={n:p for n,p in self.named_parameters() if n in self.adaptive_names};assert params.keys()==state.keys()
        with torch.no_grad():
            for n,p in params.items():p.copy_(state[n].to(p.device))
    def assert_frozen(self):
        for n,p in self.llm.named_parameters():
            if n in self.frozen_versions:assert p._version==self.frozen_versions[n] and not p.requires_grad and p.grad is None
    @torch.inference_mode()
    def predict(self,report,memory):
        self.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            inputs,ids,length,_=self.pack(report,memory);budget=min(2048,self.max_context-length);assert budget>=128
            output=self.llm.generate(**inputs,input_ids=ids,max_new_tokens=budget,do_sample=False,num_beams=1,use_cache=True,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
        assert torch.equal(output[:,:length],ids)
        tokens=output[0,length:];raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        if not len(tokens) or int(tokens[-1])!=self.tokenizer.eos_token_id:return report,dict(accepted=False,status='INCOMPLETE_EDIT_PROGRAM',raw_private=raw)
        try:
            edits=json.loads(raw);text=apply_edits(report,edits)
            assert text.strip()
        except (ValueError,TypeError,AssertionError):return report,dict(accepted=False,status='INVALID_EDIT_PROGRAM',raw_private=raw)
        return text,dict(accepted=True,status='APPLIED_EDITS' if edits else 'KEEP',edit_count=len(edits),raw_private=raw)
