"""Exercise the real ComfyUI MultiGPU CFG Split path with independent TRT engines."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', required=True, type=Path)
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.comfy_root.resolve()))
sys.argv = [sys.argv[0], '--disable-dynamic-vram', '--use-pytorch-cross-attention', '--cuda-device', 'all']
import comfy.options
comfy.options.enable_args_parsing()
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trt_reforge', root / '__init__.py', submodule_search_locations=[str(root)])
extension = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extension
spec.loader.exec_module(extension)
import torch
import folder_paths
import comfy.multigpu
from comfy import sample
from trt_reforge.trt_loader import TRTLoader
from trt_reforge.trt_diffusers.base_diffuser import TRTDiffuser

assert torch.cuda.device_count() >= 2, 'Two CUDA GPUs are required'
folder_paths.add_model_folder_path('tensorrt', str(args.engine.resolve().parent))
primary = TRTLoader.execute(model_path=args.engine.name, model_type='SDXL', sampling='auto').result[0]
assert primary.cached_patcher_init is not None
multi = comfy.multigpu.create_multigpu_deepclones(primary, 2)
replicas = [multi, *multi.get_additional_models_with_key('multigpu')]
assert len(replicas) == 2
assert {patcher.load_device for patcher in replicas} == {torch.device('cuda:0'), torch.device('cuda:1')}
assert replicas[0].model is not replicas[1].model
assert replicas[0].model.diffusion_model.engine is not replicas[1].model.diffusion_model.engine
assert replicas[0].model.diffusion_model.context is not replicas[1].model.diffusion_model.context
for patcher in replicas:
    assert patcher.model.diffusion_model.device == patcher.load_device
multi = multi.clone()
multi.match_multigpu_clones()
replicas = [multi, *multi.get_additional_models_with_key('multigpu')]

torch.manual_seed(1234)
latent_input = torch.randn(1, 4, 128, 128, dtype=torch.float16)
context = torch.randn(1, 77, 2048, dtype=torch.float16)
vector = torch.randn(1, 2816, dtype=torch.float16)
timestep = torch.tensor([500.0])


def infer(patcher):
    with torch.cuda.device(patcher.load_device):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            output = patcher.model.diffusion_model(latent_input.to(patcher.load_device),
                timestep.to(patcher.load_device), context=context.to(patcher.load_device), y=vector.to(patcher.load_device))
            output = output.float().cpu()
        stream.synchronize()
        return output


with ThreadPoolExecutor(max_workers=2) as pool:
    outputs = list(pool.map(infer, replicas))
torch.testing.assert_close(outputs[0], outputs[1], atol=0.02, rtol=0.02)

positive = [[torch.randn(1, 77, 2048), {'pooled_output': torch.randn(1, 1280)}]]
negative = [[torch.randn(1, 77, 2048), {'pooled_output': torch.randn(1, 1280)}]]
latent = torch.zeros(1, 4, 128, 128)
noise = sample.prepare_noise(latent, 1234)
sampling_args = dict(steps=4, cfg=4.0, sampler_name='euler', scheduler='normal',
                     positive=positive, negative=negative, latent_image=latent, seed=1234, disable_pbar=True)
reference = sample.sample(primary, noise, **sampling_args)

# Record actual worker dispatch, rather than inferring MultiGPU use from the
# presence of an extra patcher. This wrapper exists only in the validation tool.
dispatches = []
original_call = TRTDiffuser.__call__


def traced_call(self, latent, timestep, **kwargs):
    options = kwargs.get('transformer_options', {})
    dispatches.append((str(latent.device), tuple(options.get('cond_or_uncond', ()))))
    return original_call(self, latent, timestep, **kwargs)


TRTDiffuser.__call__ = traced_call
try:
    actual = sample.sample(multi, noise, **sampling_args)
finally:
    TRTDiffuser.__call__ = original_call
assert {device for device, _ in dispatches} == {'cuda:0', 'cuda:1'}, dispatches
assert {branch for _, branch in dispatches} == {(0,), (1,)}, dispatches
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)

report = {'result': 'PASS', 'torch': torch.__version__, 'engine': args.engine.name,
    'gpus': [torch.cuda.get_device_name(i) for i in range(2)], 'steps': 4, 'cfg': 4.0,
    'independent_models_engines_contexts': True,
    'concurrent_unet_max_abs_difference': (outputs[0] - outputs[1]).abs().max().item(),
    'sampler_max_abs_difference': (actual - reference).abs().max().item(),
    'sampler_mean_abs_difference': (actual - reference).abs().mean().item(),
    'dispatches': {f'{device}:{branch}': count for (device, branch), count in Counter(dispatches).items()}}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'multigpu-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
