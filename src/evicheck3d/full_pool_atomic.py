"""Own uniquely named atomic READY publisher, never imports legacy node policy."""
import ctypes,hashlib,json,os,sys
from pathlib import Path
from pref_inputs import need


def publish_ready(path,value,guard):
    path=Path(path);need(path.name in {f'node{n}_READY_private.json' for n in (256,512,768,1024)},'FULL_POOL_READY_NODE')
    pending=path.with_name(path.name+'.pending');raw=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
    need(len(raw)<=4<<20 and sys.platform=='linux','FULL_POOL_READY_BOUND')
    guard();fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    guard();lib=ctypes.CDLL(None,use_errno=True);need(hasattr(lib,'renameat2'),'ATOMIC_NOREPLACE_REQUIRED')
    rename=lib.renameat2;rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    need(rename(-100,os.fsencode(pending),-100,os.fsencode(path),1)==0,'FULL_POOL_READY_RENAME_FAILED')
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);os.fsync(fd);os.close(fd)
