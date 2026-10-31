"""Read-only environment report. Does not build an engine or install anything."""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

root = Path(__file__).resolve().parents[1]
if (root / '.deps').is_dir():
    sys.path.append(str(root / '.deps'))
report = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}
for name in ('torch', 'numpy', 'onnx', 'onnxscript', 'tensorrt_cu13', 'tensorrt_cu13_bindings',
             'tensorrt_cu13_libs', 'tensorrt_cu12', 'onnxruntime', 'onnxruntime-gpu'):
    try:
        report[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        report[name] = None
try:
    import torch
    report['torch_cuda'] = torch.version.cuda
    report['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name()
        report['compute_capability'] = torch.cuda.get_device_capability()
        report['free_total_vram'] = torch.cuda.mem_get_info()
except Exception as exc:
    report['torch_import_error'] = str(exc)
try:
    import tensorrt
    report['tensorrt_import_version'] = tensorrt.__version__
except Exception as exc:
    report['tensorrt_import_error'] = str(exc)
print(json.dumps(report, indent=2, ensure_ascii=False))
