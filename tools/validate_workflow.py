"""Optional real ComfyUI CLIP -> KSampler -> VAE GPU integration check."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', required=True, type=Path)
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--engine', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--deps', type=Path)
parser.add_argument('--resolution', type=int, default=1024)
parser.add_argument('--steps', type=int, default=4)
args = parser.parse_args()
if args.deps:
    sys.path.append(str(args.deps.resolve()))
sys.path.insert(0, str(args.comfy_root.resolve()))
sys.argv = [sys.argv[0], '--disable-dynamic-vram', '--use-pytorch-cross-attention']
import comfy.options
comfy.options.enable_args_parsing()
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trt_reforge', root / '__init__.py', submodule_search_locations=[str(root)])
extension = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extension
spec.loader.exec_module(extension)
import folder_paths
import torch
from comfy import sd, sample
from PIL import Image
from trt_reforge.trt_loader import TRTLoader

args.output.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path('tensorrt', str(args.engine.resolve().parent))
_, clip, vae, _ = sd.load_checkpoint_guess_config(str(args.checkpoint), output_model=False,
                                                output_clip=True, output_vae=True)


def conditioning(text):
    cond, pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
    return [[cond, {'pooled_output': pooled}]]


positive = conditioning('landscape, blue sky, mountains, lake, best quality')
negative = conditioning('worst quality, low quality, blurry, text, watermark')
patcher = TRTLoader.execute(model_path=args.engine.name, model_type='SDXL', sampling='auto').result[0]
# Exercise the clone path used by normal ComfyUI workflow modifiers.
patcher = patcher.clone()
latent = torch.zeros(1, 4, args.resolution // 8, args.resolution // 8)
noise = sample.prepare_noise(latent, 1234)
start = time.perf_counter()
with torch.inference_mode():
    samples = sample.sample(patcher, noise, steps=args.steps, cfg=4.0, sampler_name='euler', scheduler='normal',
                            positive=positive, negative=negative, latent_image=latent, seed=1234,
                            disable_pbar=True)
    if not torch.isfinite(samples).all():
        raise AssertionError('Non-finite KSampler output')
    image_tensor = vae.decode(samples)
    if not torch.isfinite(image_tensor).all():
        raise AssertionError('Non-finite VAE output')
    image = (image_tensor[0].clamp(0, 1).cpu().numpy() * 255).round().astype('uint8')
Image.fromarray(image).save(args.output / 'workflow-check.png')
report = {'checkpoint': args.checkpoint.name, 'engine': args.engine.name,
          'resolution': args.resolution, 'steps': args.steps, 'sampler': 'euler', 'cfg': 4.0, 'seed': 1234,
          'elapsed_seconds_including_vae': time.perf_counter() - start,
          'result': 'PASS: real CLIP encoding, TRTLoader, ModelPatcher.clone, ComfyUI sampling and VAE decode',
          'limitation': 'Integration check; not an image-quality or speed comparison.'}
(args.output / 'workflow-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
