"""Exact own scientific-state restore and durable dynamic-pool cursor saves."""
import hashlib,json,os,random,stat
from pathlib import Path
from pref_inputs import need,private_bytes,strict_json
from pref_checkpoint import initialization_receipt,state_hash
from replica_budget import checkpoint_state_gate,BoundedCheckpointFile
from pref_runtime import write_once,sync_directory

NODES=(1024,)


def load_parent(T,ref,guard):
    p=Path(ref['checkpoint_path']);need(p.resolve(strict=True)==p,'PARENT_CANONICAL')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb',buffering=0) as stream:
        a=os.fstat(stream.fileno());need(stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid() and a.st_nlink==1 and stat.S_IMODE(a.st_mode)==0o600 and 0<a.st_size<=1<<30,'PARENT_CHECKPOINT_METADATA')
        h=hashlib.sha256()
        while part:=stream.read(1<<20):guard();h.update(part)
        need(h.hexdigest()==ref['checkpoint_sha256'],'EXACT_PARENT_CHECKPOINT_SHA');stream.seek(0)
        state=T.load(stream,map_location='cpu',weights_only=True);b=os.fstat(stream.fileno())
    z=p.lstat();keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_uid','st_nlink')
    need(all(getattr(a,k)==getattr(b,k)==getattr(z,k) for k in keys),'PARENT_CHANGED');return state


def restore(T,model,optimizer,state,parentmeta,config,binding,initialization):
    base=config['parent_global_update'];node=binding['parent']['local_node'];ref=binding['parent']
    ready=strict_json(private_bytes(ref['ready']['path'],ref['ready']['sha256']))
    resume=strict_json(private_bytes(ref['resume']['path'],ref['resume']['sha256']))
    need(state['protocol'] in ('main96-continuation1536-own-checkpoint-v1','main96-continuation2048-own-checkpoint-v1','evicheck3d-full-train-continuation-checkpoint-v1') and
      state['protocol']==ref['checkpoint_protocol'] and state['parent_global_update']==state['global_offset']==parentmeta['parent_global_update'] and
      state['global_update']==state['parent_global_update']+node==base and state['update']==state['cursor']==node and state['triple_cursor']==node%32 and state['metadata']==parentmeta,'OWN_SCIENTIFIC_PARENT_HEADER')
    need(ready['global_update']==resume['global_update']==base and ready['local_update']==resume['node']==node and
      ready['stage_id']==ref['stage_id'] and ready['seed_group_id']==binding['seed_group_id'] and ready['arm']==binding['arm'] and
      ready['source_manifest_sha256']==binding['parent_source_manifest_sha256'] and ready['metadata_sha256']==ref['metadata']['sha256'] and
      ready['checkpoint_path']==ref['checkpoint_path'] and ready['checkpoint_sha256']==resume['checkpoint_sha256']==ref['checkpoint_sha256'] and resume['full_adaptive_optimizer_python_CPU_ownCUDA_RNG'] is True,'OWN_COMMITTED_PARENT_READY_RESUME')
    need(parentmeta['seed_group_id']==binding['seed_group_id'] and parentmeta['arm']==binding['arm'] and
      parentmeta['init_seed']==config['init_seed'] and parentmeta['train_seed']==config['train_seed'] and
      parentmeta['initialization']==initialization and parentmeta['runtime_versions']==config['runtime_versions'] and
      parentmeta['source_manifest_sha256']==binding['parent_source_manifest_sha256'],'OWN_PARENT_MODEL_SEED_RUNTIME')
    checkpoint_state_gate(T,state);params=model.trainable();adaptive=state['adaptive_state'];need(set(adaptive)==set(params),'PARENT_ADAPTIVE_KEYS')
    for name,v in adaptive.items():need(v.shape==params[name].shape and v.dtype==params[name].dtype and bool(T.isfinite(v).all()),'PARENT_ADAPTIVE_LAYOUT')
    adaptive_pin=state_hash(T,adaptive);model.restore(adaptive);model.assert_frozen()
    restored=initialization_receipt(T,model);need(restored['adaptive_sha256']==adaptive_pin and restored['bridge_buffers_sha256']==initialization['bridge_buffers_sha256'],'RESTORED_OWN_ADAPTIVE_BRIDGE')
    optimizer.load_state_dict(state['optimizer_state']);after=optimizer.state_dict()
    need(len(after['state'])==len(params) and after['param_groups']==state['optimizer_state']['param_groups'],'OWN_ADAM_GROUPS')
    for pid,values in state['optimizer_state']['state'].items():
        need(int(values['step'].item())==base,'PARENT_ADAM_GLOBAL_STEP')
        for key,value in values.items():
            got=after['state'][pid][key];need(T.equal(value.cpu(),got.cpu()) if isinstance(value,T.Tensor) else value==got,'OWN_ADAM_MOMENTS')
    need(set(state['rng_state'])=={'python','torch_cpu','own_cuda'},'OWN_RNG_FIELDS')
    random.setstate(state['rng_state']['python']);T.set_rng_state(state['rng_state']['torch_cpu']);T.cuda.set_rng_state(state['rng_state']['own_cuda'],model.device)
    prior_pool=state.get('full_pool_state')
    if prior_pool is None:
        need(config['first_full_pool_transition'] is True,'EXPLICIT_FIRST_POOL_TRANSITION');cursor=0
    else:
        need(prior_pool['manifest_sha256']==config['pool_manifest_sha256'] and config['first_full_pool_transition'] is False,'SAME_FULL_POOL_CONTINUED_MANIFEST')
        cursor=prior_pool['next_pool_cursor'];need(type(cursor) is int and cursor>=0,'RESTORED_POOL_CURSOR')
    need(base!=8192 or cursor==6144,'FIXED_9216_PARENT_CURSOR')
    need(base!=9216 or (cursor==7168 and ready['next_pool_cursor']==resume['next_pool_cursor']==cursor),'FIXED_10240_PARENT_CURSOR')
    need(base!=10240 or (cursor==8192 and ready['next_pool_cursor']==resume['next_pool_cursor']==cursor),'FIXED_11264_PARENT_CURSOR')
    return cursor,prior_pool['action_spec'] if prior_pool is not None else None


def save(T,model,optimizer,node,metadata,pool,plan,scope,guard):
    need(node in NODES,'FIXED_CHUNK_NODE');guard(saving=True);base=metadata['parent_global_update']
    path=Path(scope)/f'checkpoint_{node:03d}.pt';pending=Path(str(path)+'.pending');need(not path.exists(),'NO_CHECKPOINT_OVERWRITE')
    state=dict(protocol='evicheck3d-full-train-continuation-checkpoint-v1',parent_global_update=base,global_offset=base,global_update=base+node,update=node,cursor=node,triple_cursor=node%32,metadata=metadata,adaptive_state=model.export(),optimizer_state=optimizer.state_dict(),rng_state=dict(python=random.getstate(),torch_cpu=T.get_rng_state(),own_cuda=T.cuda.get_rng_state(model.device)),operation_seed_rule='SHA256_JSON_TRAIN_GLOBAL_UPDATE_ROLE_EDIT_FIRST8_BIG_ENDIAN_MASK63',full_pool_state=dict(manifest_sha256=metadata['pool_manifest_sha256'],next_pool_cursor=plan.cursor,action_spec=plan.action_spec))
    checkpoint_state_gate(T,state);fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:T.save(state,BoundedCheckpointFile(stream));stream.flush();os.fsync(stream.fileno())
    guard(saving=True);os.rename(pending,path);sync_directory(path.parent)
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(1<<20):guard();h.update(b)
    pin=h.hexdigest();write_once(Path(scope)/f'resume_{node:03d}_private.json',dict(protocol='evicheck3d-full-train-resume-manifest-v1',node=node,global_update=base+node,checkpoint_file=path.name,checkpoint_sha256=pin,full_adaptive_optimizer_python_CPU_ownCUDA_RNG=True,pool_manifest_sha256=metadata['pool_manifest_sha256'],next_pool_cursor=plan.cursor,resume_requires_new_root_admission=True,automatic_retry=False),guard)
    from full_pool_atomic import publish_ready
    publish_ready(Path(scope)/f'node{node}_READY_private.json',dict(protocol='evicheck3d-full-train-node-ready-v1',stage_id=metadata['global_scope_id'],seed_group_id=metadata['seed_group_id'],arm=metadata['arm'],local_update=node,global_update=base+node,checkpoint_path=str(path),checkpoint_sha256=pin,metadata_sha256=hashlib.sha256((Path(scope)/'metadata_private.json').read_bytes()).hexdigest(),source_manifest_sha256=metadata['source_manifest_sha256'],pool_manifest_sha256=metadata['pool_manifest_sha256'],next_pool_cursor=plan.cursor,development_reused=True),guard)
    return pin
