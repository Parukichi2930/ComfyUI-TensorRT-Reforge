"""Install ONNX helpers into this custom node, leaving host packages untouched."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
target = root / '.deps'
if target.exists():
    raise SystemExit(f'{target} already exists. Keep it, or rename it before a fresh install.')

installed = {dist.metadata['Name']: dist.version for dist in metadata.distributions() if dist.metadata['Name']}
# Constrain every existing host package, including torch's CUDA build suffix.
# A conflict must fail instead of installing a different version alongside it.
with tempfile.TemporaryDirectory(prefix='trt-reforge-') as temporary:
    constraints = Path(temporary) / 'host-constraints.txt'
    constraints.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(installed.items())), encoding='utf-8')
    command = [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
               '--target', str(target), '--constraint', str(constraints),
               '--requirement', str(root / 'requirements.txt'), '--disable-pip-version-check']
    print('Installing ONNX helpers inside the custom node. Host packages are not modified.', flush=True)
    subprocess.run(command, check=True)

after = {dist.metadata['Name']: dist.version for dist in metadata.distributions() if dist.metadata['Name']}
if after != installed:
    raise SystemExit('Unexpected change in host package metadata. Inspect the environment before proceeding.')
(target / 'host-versions.json').write_text(json.dumps(installed, indent=2), encoding='utf-8')
print(f'Done: {target}. Restart ComfyUI.')
