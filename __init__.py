import os
import sys

# Optional dependencies installed beside this node. Existing host packages
# retain priority, especially torch and TensorRT. No host .venv is modified.
_deps = os.path.join(os.path.dirname(__file__), ".deps")
if os.path.isdir(_deps) and _deps not in sys.path:
    sys.path.append(_deps)

from comfy_api.latest import ComfyExtension
from comfy_api.latest import io

from .trt_exporter import TRTExporter
from .trt_loader import TRTLoader

WEB_DIRECTORY = "./js"

class ComfyUITensorRTReforge(ComfyExtension):

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TRTExporter,
            TRTLoader
        ]

def comfy_entrypoint() -> ComfyUITensorRTReforge:
    return ComfyUITensorRTReforge()
