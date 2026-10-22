"""Read-only exact selected-runtime binding. No Torch import or CUDA call."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import sys

p=Path(sys.executable).resolve()
if not p.is_file() or p.stat().st_size>=32*1024**2:raise ValueError('bounded interpreter identity required')
r={'executable':sys.executable,'resolved_executable':str(p),
   'executable_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'python':sys.version,
   'packages':{n:metadata.version(n) for n in ('torch','torchvision','numpy','opencv-python')}}
print(json.dumps(r,sort_keys=True,separators=(',',':')),end='')
