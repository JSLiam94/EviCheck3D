"""Source-derived checkpoint/disk guards, used only by admitted training jobs."""
from pathlib import Path
import json
import os
import resource
import stat

ADAPTIVE_PARAMS=71537665
CHECKPOINT_CAP=1<<30
PAIR_CAP=32<<30
TENSOR_PAYLOAD_CAP=12*ADAPTIVE_PARAMS+(1<<20)


def need(ok,code):
    if not ok:raise RuntimeError(code)


def checkpoint_state_gate(T,state):
    def tensors(value):
        if isinstance(value,T.Tensor):yield value
        elif isinstance(value,dict):
            for item in value.values():yield from tensors(item)
        elif isinstance(value,(list,tuple)):
            for item in value:yield from tensors(item)
    need(sum(x.numel()*x.element_size() for x in tensors(state))<=TENSOR_PAYLOAD_CAP,
         'CHECKPOINT_TENSOR_PAYLOAD_BUDGET')
    need(len(json.dumps(state['metadata'],ensure_ascii=False,allow_nan=False).encode())<=4<<20,
         'CHECKPOINT_METADATA_BUDGET')
    for value in state['adaptive_state'].values():
        need(value.element_size()<=4,'CHECKPOINT_ADAPTIVE_DTYPE')
    need(sum(x.numel() for x in state['adaptive_state'].values())==ADAPTIVE_PARAMS,
         'CHECKPOINT_ADAPTIVE_PARAMETER_COUNT')


class BoundedCheckpointFile:
    """Refuse an oversized write without truncating a partial checkpoint."""
    def __init__(self,stream):self.stream=stream
    def write(self,value):
        need(self.stream.tell()+len(value)<=CHECKPOINT_CAP,'CHECKPOINT_FILE_CAP')
        return self.stream.write(value)
    def seek(self,*args):
        pos=self.stream.seek(*args);need(0<=pos<=CHECKPOINT_CAP,'CHECKPOINT_SEEK_CAP');return pos
    def __getattr__(self,name):return getattr(self.stream,name)


def pair_directory_budget(scope):
    root=Path(scope).parent
    need(root.resolve(strict=True)==root,'PAIR_BUDGET_CANONICAL')
    soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
    need(0<soft<=CHECKPOINT_CAP and 0<hard<=CHECKPOINT_CAP,'NATIVE_FILESIZE_HARD_CAP')
    total=0;misc=0;checkpoint_counts={'preserve_bound':0,'preserve_deranged':0}
    predictions={'preserve_bound':0,'preserve_deranged':0}
    for path in root.rglob('*'):
        try:st=path.lstat()
        except FileNotFoundError:
            # Concurrent peer pending->final/status publication may remove a
            # listed name. Resource scan is not durable artifact validation.
            continue
        need(not stat.S_ISLNK(st.st_mode),'PAIR_ARTIFACT_SYMLINK')
        if not stat.S_ISREG(st.st_mode):continue
        need(st.st_uid==os.getuid() and st.st_nlink==1 and st.st_size<=CHECKPOINT_CAP,
             'PAIR_ARTIFACT_OWNER_OR_FILE_CAP')
        total+=st.st_size;rel=path.relative_to(root);parts=rel.parts
        arm=parts[0].removeprefix('train_') if parts else ''
        if arm in checkpoint_counts and path.name.startswith('checkpoint_'):
            checkpoint_counts[arm]+=1
        elif arm in predictions and path.name.endswith('_predictions_private.jsonl'):
            predictions[arm]+=1
        elif path.name=='train_diagnostics_private.jsonl':pass
        elif path.name in {'train_preserve_bound_native_private.log','train_preserve_deranged_native_private.log'}:pass
        else:misc+=st.st_size
    # Eight logical save nodes. A concurrent snapshot may observe both a
    # pending and its newly published final name. Allow9 observed filenames
    # per arm within the ALREADY reserved18GiB, not a ninth logical node.
    need(all(n<=9 for n in checkpoint_counts.values()),'FIXED_CHECKPOINT_FILE_COUNT')
    need(predictions['preserve_bound']<=5 and predictions['preserve_deranged']<=4,
         'FIXED_PREDICTION_FILE_COUNT')
    need(misc<=1<<30 and total<=PAIR_CAP,'PAIR32_GIB_TOTAL_OR_MISC_CAP')
    return total
