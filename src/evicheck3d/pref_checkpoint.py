"""Only four fixed full-adaptive+Adam+ownRNG states, no automatic resume."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import random
from pref_inputs import need
from pref_runtime import sync_directory, write_once
from replica_budget import checkpoint_state_gate, BoundedCheckpointFile

NODES = (0, 128, 256, 512)


def state_hash(T, state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        from pref_inputs import canonical
        value = tensor.detach().cpu().contiguous()
        header = canonical([name, str(value.dtype), list(value.shape)])
        # reshape(-1) is mandatory for scalar tensors before dtype reinterpret.
        raw = value.reshape(-1).view(T.uint8).numpy().tobytes()
        result.update(len(header).to_bytes(8, 'big'))
        result.update(header)
        result.update(len(raw).to_bytes(8, 'big'))
        result.update(raw)
    return result.hexdigest()


def initialization_receipt(T, model):
    buffers = {n: p for n, p in model.named_buffers() if n.startswith('bridge.')}
    return {'adaptive_sha256': state_hash(T, model.export()),
            'bridge_buffers_sha256': state_hash(T, buffers),
            'trainable_names': sorted(model.trainable()),
            'bridge_buffer_names': sorted(buffers)}


def save(T, model, optimizer, node, metadata, scope, guard):
    need(type(node) is int and node in NODES, 'CHECKPOINT_FIXED_NODE')
    guard(saving=True)
    path = Path(scope) / f'checkpoint_{node:03d}.pt'
    pending = Path(str(path) + '.pending')
    need(not path.exists() and not path.is_symlink(), 'CHECKPOINT_OVERWRITE_FORBIDDEN')
    state = {'protocol': 'main96-dev-pref512-own-checkpoint-v1', 'update': node, 'cursor': node,
             'triple_cursor': node % 32, 'metadata': metadata,
             'adaptive_state': model.export(), 'optimizer_state': optimizer.state_dict(),
             'rng_state': {'python': random.getstate(), 'torch_cpu': T.get_rng_state(),
                           'own_cuda': T.cuda.get_rng_state(model.device)},
             'operation_seed_rule': 'SHA256_JSON_TRAIN_UPDATE_ROLE_EDIT_FIRST8_BIG_ENDIAN_MASK63'}
    checkpoint_state_gate(T, state)
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            T.save(state, BoundedCheckpointFile(stream))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    del state
    guard(saving=True)
    os.rename(pending, path)
    sync_directory(path.parent)
    pin = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(1 << 20):
            guard()
            pin.update(part)
    receipt = {'protocol': 'main96-dev-pref512-resume-manifest-v1', 'node': node,
               'checkpoint_file': path.name, 'checkpoint_sha256': pin.hexdigest(),
               'full_adaptive_optimizer_python_CPU_ownCUDA_RNG': True,
               'resume_requires_new_root_admission': True, 'automatic_retry': False}
    write_once(Path(scope) / f'resume_{node:03d}_private.json', receipt, guard)
    return pin.hexdigest()
