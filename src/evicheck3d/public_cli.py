"""Portable single-GPU entry points, separate from the admitted host dispatcher.

No model imports or data reads occur during configuration-only validation.
Public checkpoints use a distinct protocol and cannot replace research receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time

from pref_inputs import need, strict_json

PROTOCOL = 'evicheck3d-portable-checkpoint-v1'
ASSETS = ('base_model', 'ct_clip_code', 'ct_clip_checkpoint', 'ct_clip_text')


def load_config(path, mode):
    config_path = Path(path).resolve(strict=True)
    c = strict_json(config_path.read_bytes())
    common = {'schema', 'assets', 'output_dir', 'wall_seconds', 'execution_environment'}
    extra = ({'pool', 'init_seed', 'train_seed', 'updates', 'lambda_pref', 'resume'}
             if mode == 'train' else {'input_jsonl', 'checkpoint', 'checkpoint_sha256'})
    need(type(c) is dict and set(c) == common | extra, 'CONFIG_FIELDS')
    need(c['schema'] == 'evicheck3d-' + mode + '-v1', 'CONFIG_SCHEMA')
    need(c['execution_environment'] == 'dedicated_single_gpu', 'DEDICATED_GPU_REQUIRED')
    need(type(c['wall_seconds']) is int and 1 <= c['wall_seconds'] <= 86400, 'WALL_SECONDS')
    need(type(c['assets']) is dict and set(c['assets']) == set(ASSETS), 'ASSET_FIELDS')

    def local(value):
        need(type(value) is str and bool(value.strip()), 'LOCAL_PATH')
        p = Path(value).expanduser()
        return str((config_path.parent / p).resolve() if not p.is_absolute() else p.resolve())

    c['assets'] = {k: local(v) for k, v in c['assets'].items()}
    c['output_dir'] = local(c['output_dir'])
    if mode == 'train':
        need(all(type(c[k]) is int and 0 <= c[k] < 2**63 for k in ('init_seed', 'train_seed')), 'SEED')
        need(type(c['updates']) is int and c['updates'] == 1024, 'FIXED_1024_UPDATE_CHUNK')
        need(type(c['lambda_pref']) in (int, float) and c['lambda_pref'] in (0, .1), 'LAMBDA_PREF')
        need(type(c['pool']) is dict and set(c['pool']) == {'manifest_path', 'manifest_sha256'}, 'POOL_FIELDS')
        c['pool']['manifest_path'] = local(c['pool']['manifest_path'])
        pin(c['pool']['manifest_sha256'])
        if c['resume'] is not None:
            need(type(c['resume']) is dict and set(c['resume']) == {'path', 'sha256'}, 'RESUME_FIELDS')
            c['resume']['path'] = local(c['resume']['path']); pin(c['resume']['sha256'])
    else:
        c['input_jsonl'] = local(c['input_jsonl'])
        c['checkpoint'] = local(c['checkpoint']); pin(c['checkpoint_sha256'])
    return c


def pin(value):
    need(type(value) is str and len(value) == 64 and all(x in '0123456789abcdef' for x in value), 'SHA256')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def source_binding():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def training_binding(c):
    return {k: c[k] for k in ('assets', 'pool', 'init_seed', 'train_seed', 'lambda_pref')}


def prepare(c, mode):
    # The public runner is not an alternative launch path on the coordinated
    # research host. It never imports SSH, registration, or recovery controllers.
    need(hashlib.sha256(socket.gethostname().encode()).hexdigest() != '00c1519b42a18cdd029e28490bcc39ad5c6220c9a971993e5eb6aa93b48d8811', 'USE_SEPARATELY_ADMITTED_RESEARCH_DISPATCHER')
    need(sys.platform == 'linux', 'LINUX_CUDA_REQUIRED')
    need(os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() not in ('', '-1')
         and ',' not in os.environ['CUDA_VISIBLE_DEVICES'], 'EXPLICIT_SINGLE_GPU_VISIBILITY')
    for key, value in c['assets'].items():
        p = Path(value)
        need(p.is_file() if key == 'ct_clip_checkpoint' else p.is_dir(), 'LOCAL_ASSET_MISSING')
        os.environ['EVICHECK3D_' + key.upper()] = value
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.umask(0o077)
    out = Path(c['output_dir'])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_json(out / 'RUN_ONCE.json', {'mode': mode, 'automatic_retry': False})
    return out


def stack(c):
    import torch as T
    need(T.cuda.is_available() and T.cuda.device_count() == 1, 'ONE_VISIBLE_CUDA_DEVICE')
    need(T.cuda.is_bf16_supported(), 'BFLOAT16_REQUIRED')
    from main96_model import verify_source_imports
    verify_source_imports()
    from pref_model import ContentPreferenceRepair
    from frozen_ct_encoder_cache_v1 import CTEncoder
    return T, ContentPreferenceRepair, CTEncoder


def runtime_guard(c, output):
    import resource
    deadline = time.monotonic() + c['wall_seconds']

    def guard():
        need(time.monotonic() < deadline and not (output / 'STOP_REQUEST').exists(), 'WALL_OR_STOP')
        with open('/proc/meminfo') as f:
            available = next(int(line.split()[1]) * 1024 for line in f if line.startswith('MemAvailable:'))
        need(available >= 192 * (1 << 30), 'RUNTIME_RAM_192_GIB')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 96 * (1 << 30), 'WORKER_RSS_96_GIB')
        need(shutil.disk_usage(output).free >= 328 * (1 << 30)
             and shutil.disk_usage('/').free >= 64 * (1 << 30), 'DATA_ROOT_FREE_SPACE')
    guard()
    return guard


def load_checkpoint(T, path, sha256):
    with open(path, 'rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
        need(h.hexdigest() == sha256, 'CHECKPOINT_SHA256')
        f.seek(0); state = T.load(f, map_location='cpu', weights_only=True)
    need(state['protocol'] == PROTOCOL and state['source_binding'] == source_binding(), 'CHECKPOINT_PROTOCOL_SOURCE')
    return state


def train(c):
    out = prepare(c, 'train')
    guard = runtime_guard(c, out)
    with open('/proc/meminfo') as f:
        available = next(int(line.split()[1]) * 1024 for line in f if line.startswith('MemAvailable:'))
    need(available >= 580 * (1 << 30), 'INITIAL_RAM_580_GIB')
    T, Model, Encoder = stack(c)
    random.seed(c['init_seed']); T.manual_seed(c['init_seed'])
    model = Model()
    from full_pool_adapter import FullPoolIndex
    from full_pool_tokens import FullPoolPlan
    from full_pool_features import OnlineFrozenFeatures
    from full_pool_step import train_step
    pool = FullPoolIndex(c['pool'], guard)
    params = model.trainable()
    groups = [dict(params=[p for n, p in params.items() if n.startswith(prefix)], lr=lr)
              for prefix, lr in (('llm.', 2e-5), ('bridge.', 1e-4))]
    optimizer = T.optim.AdamW(groups, weight_decay=.01, betas=(.9, .999), eps=1e-8,
                             amsgrad=False, maximize=False, capturable=False,
                             differentiable=False, foreach=None, fused=None)
    base = cursor = 0; action = None; state = None
    if c['resume']:
        state = load_checkpoint(T, c['resume']['path'], c['resume']['sha256'])
        need(state['training_binding'] == training_binding(c), 'RESUME_TRAINING_BINDING')
        model.restore(state['adaptive_state']); optimizer.load_state_dict(state['optimizer_state'])
        base, cursor, action = state['global_update'], state['next_pool_cursor'], state['action_spec']
    plan = FullPoolPlan(model, pool, cursor, guard, restored_action_spec=action)
    features = OnlineFrozenFeatures(T, model.device, pool, guard, Encoder)
    if state:
        random.setstate(state['rng_state']['python']); T.set_rng_state(state['rng_state']['torch_cpu'])
        T.cuda.set_rng_state(state['rng_state']['own_cuda'], model.device)
        del state
    else:
        random.seed(c['train_seed']); T.manual_seed(c['train_seed'])
    progress = {'optimizer_update_inflight': None, 'optimizer_updates': base}
    try:
        for update in range(base + 1, base + c['updates'] + 1):
            guard(); features.prepare(plan.template_at(plan.cursor))
            train_step(T, model, optimizer, plan, update, features, c['train_seed'],
                       c['lambda_pref'], guard, progress)
            plan.committed()
        guard()
        final = out / 'checkpoint.pt'; pending = out / 'checkpoint.pt.pending'
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            T.save(dict(protocol=PROTOCOL, source_binding=source_binding(),
                        training_binding=training_binding(c), global_update=update,
                        next_pool_cursor=plan.cursor, action_spec=plan.action_spec,
                        adaptive_state=model.export(), optimizer_state=optimizer.state_dict(),
                        rng_state=dict(python=random.getstate(), torch_cpu=T.get_rng_state(),
                                       own_cuda=T.cuda.get_rng_state(model.device))), f)
            f.flush(); os.fsync(f.fileno())
        guard(); pending.rename(final)
        write_json(out / 'COMPLETE.json', {'checkpoint_sha256': digest(final),
                   'global_update': update, 'next_pool_cursor': plan.cursor, 'automatic_retry': False})
    finally:
        pool.db.close()


def infer(c):
    out = prepare(c, 'infer'); guard = runtime_guard(c, out)
    T, Model, Encoder = stack(c)
    state = load_checkpoint(T, c['checkpoint'], c['checkpoint_sha256'])
    need(state['training_binding']['assets'] == c['assets'], 'INFERENCE_ASSET_BINDING')
    model = Model(); model.restore(state['adaptive_state']); del state
    encoder = Encoder('cpu'); encoder.requires_grad_(False); encoder.eval()
    with open(c['input_jsonl'], encoding='utf-8') as rows:
        fd = os.open(out / 'predictions.private.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as predictions:
            for line in rows:
                guard(); row = strict_json(line)
                need(type(row) is dict and set(row) == {'report', 'image_path'}, 'INFERENCE_FIELDS_NO_REFERENCE')
                need(type(row['report']) is str and bool(row['report']), 'REPORT_TEXT')
                image = Path(row['image_path'])
                need(image.is_absolute() and image.is_file(), 'ABSOLUTE_CT_PATH')
                encoder.to(model.device); encoder.device = model.device
                try:
                    with T.no_grad(), T.autocast('cuda', dtype=T.bfloat16):
                        features = encoder(str(image)).detach().float().cpu().contiguous()
                finally:
                    encoder.to('cpu'); encoder.device = T.device('cpu'); T.cuda.empty_cache()
                guard(); text, audit = model.predict(row['report'], features)
                predictions.write(json.dumps({'repaired_report': text, 'audit': audit}, ensure_ascii=False) + '\n')
                predictions.flush()
            os.fsync(predictions.fileno())
    write_json(out / 'COMPLETE.json', {'completed': True, 'automatic_retry': False})


def main(mode):
    parser = argparse.ArgumentParser(description='EviCheck3D ' + mode + ' (local dedicated Linux/CUDA only)')
    parser.add_argument('--config', required=True, help='Local JSON config; relative paths resolve beside it')
    parser.add_argument('--check-config', action='store_true', help='Validate schema only, without reading datasets or loading models')
    args = parser.parse_args()
    try:
        c = load_config(args.config, mode)
        if args.check_config:
            print('CONFIG_SCHEMA_OK (assets, data, hardware and model execution not checked)')
            return
        (train if mode == 'train' else infer)(c)
        print('COMPLETE (private outputs remain in the configured output directory)')
    except Exception as exc:
        # Do not print private inputs, model output, or underlying library traces.
        safe = str(exc)
        code = safe if safe.isupper() and safe.replace('_', '').isalnum() and len(safe) < 96 else type(exc).__name__
        print('FAILED: ' + code + '; no automatic retry', file=sys.stderr)
        raise SystemExit(1)
