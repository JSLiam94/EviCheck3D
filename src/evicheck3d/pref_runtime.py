"""Bounded native guard and private writes. Root owns launch, lifecycle, once,
leases and GLOBAL growth registration. No querying other GPUs or killing jobs.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import stat
import time
from pref_inputs import canonical, need, private_bytes, strict_json

GIB = 1 << 30


def source_closure(package, pin):
    package = Path(package)
    need(package.resolve(strict=True) == package, 'SOURCE_CANONICAL')
    raw = (package / 'SOURCE_SHA256.json').read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin, 'SOURCE_MANIFEST_PIN')
    manifest = strict_json(raw)
    need(manifest['protocol'] == 'main96-dev-pref512-source-v1', 'SOURCE_PROTOCOL')
    need({p.name for p in package.iterdir()} == set(manifest['files']) | {'SOURCE_SHA256.json'},
         'SOURCE_EXACT_CLOSURE')
    for name, sha in manifest['files'].items():
        need(Path(name).name == name, 'SOURCE_COMPONENT')
        p = package / name
        st = p.lstat()
        need(p.resolve() == p and stat.S_ISREG(st.st_mode) and st.st_nlink == 1
             and hashlib.sha256(p.read_bytes()).hexdigest() == sha, 'SOURCE_FILE_PIN')
    return manifest['files']


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value, guard):
    path = Path(path)
    raw = canonical(value) + b'\n'
    need(len(raw) <= 4 << 20, 'METADATA_FILE_CAP')
    guard()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    sync_directory(path.parent)
    guard()
    return hashlib.sha256(raw).hexdigest()


def scan_bytes(root):
    """Owned output stat scan only, never inspect private record contents."""
    total = 0
    for p in Path(root).rglob('*'):
        try:
            st = p.lstat()
        except FileNotFoundError:
            continue  # Peer's pending->final publish race, never data validation.
        need(not stat.S_ISLNK(st.st_mode), 'OUTPUT_SYMLINK')
        if stat.S_ISREG(st.st_mode):
            need(st.st_uid == os.getuid() and st.st_nlink == 1, 'OUTPUT_OWNER_LINKS')
            total += st.st_size
    return total


class NativeGuard:
    def __init__(self, config, binding, scope, owner_check):
        self.config, self.binding, self.scope = config, binding, Path(scope)
        self.owner_check = owner_check
        self.deadline = binding['deadline_monotonic']
        self.T = None
        self.last = 0.0
        self.stage_counts = {}
        need(hashlib.sha256(socket.gethostname().encode()).hexdigest() == '00c1519b42a18cdd029e28490bcc39ad5c6220c9a971993e5eb6aa93b48d8811', 'ADMITTED_HOST_ONLY')
        need(len(binding['cpu_set']) == len(set(binding['cpu_set'])) == 2
             and os.sched_getaffinity(0) == set(binding['cpu_set']), 'TWO_CPU_AFFINITY')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            need(os.environ.get(name) == '2', 'TWO_THREADS_ENV')
        need(os.environ.get('TOKENIZERS_PARALLELISM') == 'false'
             and os.environ.get('HF_HUB_OFFLINE') == '1'
             and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'OFFLINE_NO_TOKENIZER_WORKERS')
        for name in ('TMPDIR', 'HF_HOME', 'TORCH_HOME'):
            need(os.environ.get(name, '').startswith('/data/'), 'NO_ROOT_CACHE')
        st = self.scope.stat()
        need(self.scope.resolve(strict=True) == self.scope and st.st_uid == os.getuid()
             and stat.S_IMODE(st.st_mode) == 0o700, 'PRIVATE_SCOPE')
        # No file can consume the checkpoint or misc allowance in one unbounded write.
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        need(0 < soft <= GIB and 0 < hard <= GIB, 'OWNER_FILESIZE_RLIMIT_REQUIRED')
        self(startup=True, force=True)

    def budget(self, extra_misc=0):
        total = misc = 0
        checkpoints = []
        predictions = []
        for p in self.scope.rglob('*'):
            try:
                st = p.lstat()
            except FileNotFoundError:
                continue
            need(not stat.S_ISLNK(st.st_mode), 'NATIVE_OUTPUT_SYMLINK')
            if not stat.S_ISREG(st.st_mode):
                continue
            need(st.st_uid == os.getuid() and st.st_nlink == 1 and st.st_size <= GIB,
                 'NATIVE_FILE_OWNER_CAP')
            total += st.st_size
            if p.name.startswith('checkpoint_'):
                checkpoints.append(p.name)
                need(p.name in {f'checkpoint_{n:03d}.pt{s}'
                                for n in (0, 128, 256, 512) for s in ('', '.pending')},
                     'EXTRA_CHECKPOINT_FORBIDDEN')
            else:
                misc += st.st_size
                if p.name.endswith('_predictions_private.jsonl'):
                    predictions.append(p.name)
        # Snapshot of own worker is sequential; no fifth pending/final logical save.
        need(len(checkpoints) <= 4 and len(predictions) <= 5 and misc + extra_misc <= GIB
             and total + extra_misc <= 6 * GIB, 'NATIVE_6GIB_FOUR_NODE_BUDGET')
        return total

    def __call__(self, *, startup=False, saving=False, force=False):
        now = time.monotonic()
        need(now < self.deadline, 'INCOMPLETE_DEADLINE')
        need(not (self.scope / 'STOP_REQUEST').exists() and
             not (self.scope / 'STOP_REQUEST').is_symlink(), 'OWN_STOP_REQUEST')
        if not (force or startup or saving or now - self.last >= 0.5):
            return
        self.owner_check()
        need(os.sched_getaffinity(0) == set(self.binding['cpu_set']), 'CPU_AFFINITY_CHANGED')
        self.budget()
        growth = self.config['growth']
        root = Path(growth['global_scope_root'])
        need(root.resolve(strict=True) == root and self.scope.is_relative_to(root), 'GROWTH_SCOPE_ROOT')
        actual = scan_bytes(root)
        need(actual <= 50 * GIB, 'GLOBAL50_GIB_CAP')
        ledger = strict_json(private_bytes(growth['ledger_path'], limit=65536))
        need(ledger['protocol'] == 'main96-dev-pref-global-growth-ledger-v1'
             and ledger['scope_id'] == self.config['global_scope_id']
             and ledger['state'] == 'ACTIVE_REGISTERED'
             and ledger['reserved_bytes'] == 50 * GIB
             and ledger['global_scope_root'] == str(root), 'GLOBAL_RESERVATION_REQUIRED')
        other = ledger['other_outstanding_growth_bytes']
        need(type(other) is int and other >= 0, 'OUTSTANDING_GROWTH_SCHEMA')
        # Root must update other reservations under its canonical admission lock.
        # All8 workers observe the SAME50GiB scope reservation, never eight extra50GiB.
        remaining = 50 * GIB - actual
        floor = 304 if startup else 302 if saving else 300
        need(shutil.disk_usage('/data').free >= max(floor * GIB, 300 * GIB + remaining + other + 4 * GIB)
             and shutil.disk_usage('/').free >= 64 * GIB, 'DISK_GLOBAL_GROWTH_GATE')
        import psutil
        own = psutil.Process()
        tree = [own] + own.children(recursive=True)
        need(max(sum(p.memory_info().rss for p in tree),
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) <= 96 * GIB,
             'OWN_RSS96_GIB_CAP')
        need(all(set(p.cpu_affinity()) <= set(self.binding['cpu_set']) for p in tree), 'OWN_CHILD_CPU_ESCAPE')
        # Shared832GiB gate is measured once by root BEFORE all8 launches;
        # reapplying832 independently after a peer loads would double-reserve RAM.
        if self.T is not None:
            T = self.T
            need(T.cuda.device_count() == 1, 'ONE_VISIBLE_UUID')
            need(max(T.cuda.memory_reserved(), T.cuda.max_memory_reserved(),
                     T.cuda.device_memory_used(0)) <= 40 * GIB, 'GPU40_GIB_CAP')
        self.last = time.monotonic()
