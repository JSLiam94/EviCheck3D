"""Portable path settings; legacy host dispatch is deliberately unavailable."""
from pathlib import Path
import json
import os
R = Path(os.environ.get('EVICHECK3D_DATA_ROOT', '.evicheck3d')).resolve()
C = R / 'control'
O = R / 'outputs'
OLD = C
RAD = Path(os.environ.get('EVICHECK3D_RADFM_ROOT', str(R / 'RadFM')))
def guard():
    raise RuntimeError('Use the configured public CLI or the separately admitted research dispatcher')
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)
