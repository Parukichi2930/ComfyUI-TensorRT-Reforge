"""GPU validation without installing or updating any host packages.

Run with the ComfyUI interpreter. --deps is an optional workspace-only directory
of additional libraries; existing site-packages take priority.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--deps', type=Path)
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--resolution', type=int, default=512)
parser.add_argument('--reuse-engine', type=Path)
args = parser.parse_args()
if args.deps:
    sys.path.append(str(args.deps.resolve()))
sys.path.insert(0, str(args.comfy_root.resolve()))
sys.argv = [sys.argv[0], '--disable-dynamic-vram', '--use-pytorch-cross-attention']
import comfy.options
comfy.options.enable_args_parsing()
import folder_paths
args.output.mkdir(parents=True, exist_ok=True)
folder_paths.set_output_directory(str(args.output.resolve()))
folder_paths.set_temp_directory(str((args.output / 'temp').resolve()))
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trt_reforge', root / '__init__.py', submodule_search_locations=[str(root)])
extension = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extension
spec.loader.exec_module(extension)
import torch
import tensorrt as trt
import onnx
from trt_reforge.trt_exporter import TRTExporter, _build_tensorrt_engine, _prepare_model_for_onnx_export
from trt_reforge.trt_diffusers.base_diffuser import TRTDiffuser
from trt_reforge.trt_loader import TRTLoader

report = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'tensorrt': trt.__version__,
          'onnx': onnx.__version__, 'gpu': torch.cuda.get_device_name(), 'tests': []}


def finish():
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if args.checkpoint:
    folder_paths.add_model_folder_path('checkpoints', str(args.checkpoint.resolve().parent))
    if not args.reuse_engine:
        TRTExporter.execute(
            model_name=args.checkpoint.name, filename_prefix='tensorrt/', enable_lora=False,
            opt_batch_size=1, min_batch_size=1, max_batch_size=1,
            opt_width=args.resolution, min_width=args.resolution, max_width=args.resolution,
            opt_height=args.resolution, min_height=args.resolution, max_height=args.resolution,
            opt_context_mult=1, min_context_mult=1, max_context_mult=3,
            num_video_frames=1, precision='fp16', onnx_exporter='auto', workspace_gib=1)
        report['tests'].append('SDXL checkpoint export and engine build')
    # Compare the compiled UNet against the same checkpoint with nonzero inputs.
    from trt_reforge.trt_exporter import _load_model_from_basename
    from comfy import model_management
    from trt_reforge.definitions import trt_runtime
    engine_path = args.reuse_engine or next((args.output / 'tensorrt').glob('*.engine'))
    _, patcher = _load_model_from_basename(args.checkpoint.name, torch.float16)
    model_management.load_models_gpu([patcher], force_full_load=True)
    torch.manual_seed(1234)
    latent = torch.randn(1, 4, args.resolution // 8, args.resolution // 8, device='cuda', dtype=torch.float16)
    timestep = torch.tensor([500.0], device='cuda')
    vector = torch.randn(1, 2816, device='cuda', dtype=torch.float16)
    contexts = {length: torch.randn(1, length, 2048, device='cuda', dtype=torch.float16)
                for length in (77, 231)}
    references = {}
    with torch.inference_mode(), _prepare_model_for_onnx_export():
        for length, context in contexts.items():
            references[length] = patcher.model.diffusion_model(latent, timestep, context, vector).float().cpu()
    model_management.unload_all_models()
    del patcher
    gc.collect()
    model_management.soft_empty_cache()
    with engine_path.open('rb') as source:
        diffuser = TRTDiffuser(trt_runtime.deserialize_cuda_engine(source.read()))
    report['unet_error_by_context_length'] = {}
    for length, context in contexts.items():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Inputs were created on the default stream; establish producer order.
            stream.wait_stream(torch.cuda.default_stream())
            compiled = diffuser(latent, timestep, context=context, y=vector).float()
        stream.synchronize()
        compiled = compiled.cpu()
        if not torch.isfinite(compiled).all():
            raise AssertionError('Non-finite TensorRT UNet output')
        reference = references[length]
        error = (compiled - reference).abs()
        report['unet_error_by_context_length'][length] = {'max_abs': error.max().item(),
            'mean_abs': error.mean().item(), 'rmse': error.square().mean().sqrt().item()}
        torch.testing.assert_close(compiled, reference, atol=0.08, rtol=0.08)
    report['tests'].append('SDXL UNet vs PyTorch: context 77/231 on nondefault streams (one latent/timestep)')
    report['resolution'] = args.resolution
    finish()
    raise SystemExit()


class ToyUNet(torch.nn.Module):
    def forward(self, latent, timestep, context, vector_cond):
        return latent[:, :2] * 2 + timestep[:, None, None, None] * 0.001 + context.mean((1, 2))[:, None, None, None] + vector_cond.mean(1)[:, None, None, None]


device = torch.device('cuda:0')
model = ToyUNet().to(device).eval()
shapes_min = ((1, 4, 8, 8), (1,), (1, 77, 2048), (1, 2816))
shapes_max = ((1, 4, 16, 16), (1,), (1, 231, 2048), (1, 2816))
names = ['latent', 'timestep', 'context', 'vector_cond']
inputs = tuple(torch.zeros(shape, device=device, dtype=torch.float32 if name == 'timestep' else torch.float16)
               for name, shape in zip(names, shapes_min))
onnx_path = args.output / 'toy.onnx'
from trt_reforge.trt_compat import legacy_dynamic_axes
torch.onnx.export(model, inputs, str(onnx_path), dynamo=False, opset_version=18,
                  input_names=names, output_names=['output'], external_data=True,
                  dynamic_axes=legacy_dynamic_axes(names, shapes_min, shapes_max, ['output']))
onnx.checker.check_model(str(onnx_path))
engine_path = args.output / 'toy.engine'
_build_tensorrt_engine(str(onnx_path), names, shapes_min, shapes_min, shapes_max, str(engine_path), False)
from trt_reforge.definitions import trt_runtime
with engine_path.open('rb') as source:
    diffuser = TRTDiffuser(trt_runtime.deserialize_cuda_engine(source.read()))
for shape, length, batch in ((8, 77, 2), (16, 231, 1), (8, 154, 2)):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = (torch.randn(batch, 4, shape, shape, device=device, dtype=torch.float16).transpose(2, 3),
                  torch.full((batch,), 500., device=device),
                  torch.randn(batch, length, 2048, device=device, dtype=torch.float16),
                  torch.randn(batch, 2816, device=device, dtype=torch.float16))
        expected = model(*values)
        actual = diffuser(values[0], values[1], context=values[2], y=values[3])
        # Consume immediately on the same nondefault stream to catch ordering bugs.
        consumed = actual * 3
    stream.synchronize()
    torch.testing.assert_close(consumed, expected * 3, atol=0.03, rtol=0.02)
report['tests'].extend(['TRT11 strong typing + legacy ONNX opset18',
                        'nondefault streams / noncontiguous input / CFG batch split',
                        'dynamic spatial/context shapes; output channels differ from input'])
try:
    diffuser(values[0], values[1], context=values[2][:, :10], y=values[3])
    raise AssertionError('Out-of-profile input was accepted')
except ValueError:
    report['tests'].append('out-of-profile input rejected')

# A real ComfyUI V3 node and ModelPatcher clone, including NoobAI-style sampling.
metadata = {'schema_version': 1, 'model_family': 'SDXL', 'prediction_type': 'V_PREDICTION',
            'sampling_settings': {'zsnr': True}, 'unet_config': {'adm_in_channels': 2816, 'in_channels': 4}}
Path(str(engine_path) + '.json').write_text(json.dumps(metadata), encoding='utf-8')
folder_paths.add_model_folder_path('tensorrt', str(args.output.resolve()))
node_output = TRTLoader.execute(model_path='toy.engine', model_type='SDXL', sampling='auto')
patcher = node_output.result[0]
assert patcher.model.model_type.name == 'V_PREDICTION'
assert patcher.model.model_sampling.zsnr
cloned = patcher.clone()
assert cloned.model is patcher.model
TRTExporter.define_schema()
TRTLoader.define_schema()
report['tests'].extend(['ComfyUI V3 schemas / loader / ModelPatcher.clone fast_disk API',
                        'V_PREDICTION + zero-terminal-SNR metadata restored'])
finish()
