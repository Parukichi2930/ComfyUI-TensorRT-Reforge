from typing import Any, Optional
from typing_extensions import override, cast, assert_never, Unpack

import os
import time
import json
import gc
import hashlib
from contextlib import contextmanager
import tqdm
import numpy as np

import torch
from torch import nn
import tensorrt as trt
import onnx

import comfy.cli_args
import comfy.model_patcher
import comfy.quant_ops
import comfy.ops
import comfy.ldm.modules.attention
from comfy_api.latest import io
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.ldm.anima.model import LLMAdapter
from comfy import model_base, model_management, sd
import folder_paths

from .trt_utils import ModelBundle, ensure_temp_or_output_path
from .definitions import WeightsNameMap, WeightsShapeMap, SupportedModelType, TRTSpec, trt_logger
from .trt_compat import network_creation_flags, legacy_dynamic_axes, make_export_metadata


class TRTExporter(io.ComfyNode):
    """
    Exports a TensorRT engine file and its associated ONNX file (if required) from a given model patcher and specifications.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_name_options = _get_model_name_options()
        return io.Schema(
            node_id="TensorRTExporterNode",
            display_name="TensorRT Exporter Reforge",
            category="TensorRT",
            inputs=[
                io.Combo.Input(
                    id="model_name",
                    display_name="Model Name",
                    options=model_name_options,
                    default=model_name_options[0] if model_name_options else ""
                ),
                io.Int.Input(id="opt_width", display_name="Opt Width", default=512, min=1), #Width First for comfyui
                io.Int.Input(id="opt_height", display_name="Opt Height", default=512, min=1),
                io.Int.Input(id="opt_batch_size", display_name="Opt Batch Size", default=1, min=1),

                io.Boolean.Input(id="enable_lora", display_name="Enable LoRA (Experimental🧪)", default=False),
                io.Combo.Input(id="precision", options=["fp16", "fp32"], default="fp16"),
                io.Combo.Input(id="onnx_exporter", options=["auto", "legacy", "dynamo"], default="auto"),
                io.Int.Input(id="workspace_gib", default=1, min=1, max=16),
                io.Int.Input(id="opt_context_mult", display_name="Opt Context Multiplier", default=1, min=1),
                io.Int.Input(id="num_video_frames", display_name="Num Video Frames", default=1, min=1),
                io.String.Input(id="filename_prefix", display_name="Filename Prefix", default="tensorrt/"),

                io.Int.Input(id="min_width", display_name="Min Width", default=0, min=0),
                io.Int.Input(id="max_width", display_name="Max Width", default=0, min=0),

                io.Int.Input(id="min_height", display_name="Min Height", default=0, min=0),
                io.Int.Input(id="max_height", display_name="Max Height", default=0, min=0),

                io.Int.Input(id="min_batch_size", display_name="Min Batch Size", default=0, min=0),
                io.Int.Input(id="max_batch_size", display_name="Max Batch Size", default=0, min=0),

                io.Int.Input(id="min_context_mult", display_name="Min Context Multiplier", default=0, min=0),
                io.Int.Input(id="max_context_mult", display_name="Max Context Multiplier", default=0, min=0),
            ],
            is_output_node=True
        )

    @classmethod
    @override
    def execute(cls, **kwargs: Unpack[TRTSpec]) -> io.NodeOutput:
        _validate_export_env()

        model_name = kwargs["model_name"]

        spec = _adjust_spec_range(kwargs)
        precision = spec.get("precision", "fp16")
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
        if int(trt.__version__.split('.')[0]) >= 11 and spec["enable_lora"]:
            raise ValueError("TensorRT 11 SDXL compatibility mode requires enable_lora=False. "
                             "Refit/LoRA conversion has not been validated for this mode.")
        model_source, model_patcher = _load_model_from_basename(model_name, dtype=dtype)

        del kwargs

        min_batch_size, opt_batch_size, max_batch_size = spec["min_batch_size"], spec["opt_batch_size"], spec["max_batch_size"]
        min_height, opt_height, max_height = spec["min_height"], spec["opt_height"], spec["max_height"]
        min_width, opt_width, max_width = spec["min_width"], spec["opt_width"], spec["max_width"]
        min_context_mult, opt_context_mult, max_context_mult = spec["min_context_mult"], spec["opt_context_mult"], spec["max_context_mult"]

        model_management.unload_all_models()
        model_management.load_models_gpu([model_patcher], force_full_load=True) # pyright: ignore[reportUnknownMemberType]
        
        diffusion_model = cast(model_base.BaseModel, model_patcher.model) # pyright: ignore[reportUnknownMemberType]
        diffuser = diffusion_model.diffusion_model

        model_config: Any = diffusion_model.model_config
        unet_config: Optional[dict[str, Any]] = None
        if hasattr(model_config, "unet_config"):
            unet_config = cast(dict[str, Any], model_config.unet_config)
        
        model_type = SupportedModelType.from_instance(diffusion_model)

        context_len_min, context_len, context_dim, dtype = _get_context_features(model_type, unet_config)
        if model_type == SupportedModelType.SDXL:
            dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
            if context_dim != 2048 or diffusion_model.adm_channels != 2816 or unet_config.get("in_channels", 4) != 4:
                raise ValueError("This compatibility mode supports standard SDXL base UNets (4/2048/2816).")
            for key in ("min_width", "opt_width", "max_width", "min_height", "opt_height", "max_height"):
                if spec[key] < 64 or spec[key] % 64:
                    raise ValueError(f"{key} must be a positive multiple of 64 for SDXL export.")
            diffuser.to(dtype=dtype)

        use_rope3d = unet_config.get("pos_emb_cls", "None") == "rope3d" if unet_config is not None else False
        num_video_frames = spec["num_video_frames"]
        if model_type == SupportedModelType.SVD:
            min_batch_size *= num_video_frames
            max_batch_size *= num_video_frames

        y_dim = cast(int, diffusion_model.adm_channels) # pyright: ignore[reportUnknownMemberType]


        inputs, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, input_names, output_names, dynamic_shapes = _build_onnx_tracing_input(
            unet_config, model_type,
            min_batch_size, opt_batch_size, max_batch_size,
            min_context_mult, opt_context_mult, max_context_mult,
            min_height, opt_height, max_height,
            min_width, opt_width, max_width,
            context_len_min, context_len, context_dim, dtype,
            use_rope3d, num_video_frames, y_dim,
        )

        is_svd = unet_config.get("use_temporal_resblock", False) == True if unet_config is not None else False
        transformer_options = cast(dict[str, Any], model_patcher.model_options['transformer_options'].copy()) # pyright: ignore[reportUnknownMemberType]

        tracing_model = _build_onnx_tracing_model(is_svd, diffuser, transformer_options, num_video_frames)

        temp_dir = folder_paths.get_temp_directory()
        output_onnx = os.path.normpath(os.path.join(temp_dir, f"trt_{time.time()}", "model.onnx"))
        output_onnx = ensure_temp_or_output_path(output_onnx)
        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

        exporter = spec.get("onnx_exporter", "auto")
        use_dynamo = exporter == "dynamo" or (exporter == "auto" and model_type != SupportedModelType.SDXL)
        if not use_dynamo and model_type not in (SupportedModelType.SDXL, SupportedModelType.SD15):
            raise ValueError("The legacy ONNX exporter is supported only for SDXL/SD15.")
        print(f"[TensorRT] Exporting ONNX Model (dynamo={use_dynamo}, dtype={dtype}) ...")
        # Anima is supported in opset 25
        
        match model_type:
            case SupportedModelType.Anima:
                opset_version = 25
            case _:
                opset_version = 18

        enable_lora = spec["enable_lora"]

        metadata = make_export_metadata(diffusion_model, model_type, spec, input_names,
                                        inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max)
        export_options = ({"dynamic_shapes": dynamic_shapes, "optimize": not enable_lora}
                          if use_dynamo else {"dynamic_axes": legacy_dynamic_axes(
                              input_names, inputs_shapes_min, inputs_shapes_max, output_names)})
        with _prepare_model_for_onnx_export(), torch.inference_mode():
            torch.onnx.export( # pyright: ignore[reportUnknownMemberType]
                tracing_model.eval(),
                inputs,
                output_onnx,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                **export_options,
                external_data=True,
                dynamo=use_dynamo,
                do_constant_folding=False if enable_lora else True, # Constant folding can interfere with LoRA weight mapping, so disable it when LoRA is enabled
            )

        if enable_lora:
            print("[TensorRT] Analyzing model for LoRA ...")
            weights_name_mapping, weights_shape_mapping = _get_weights_mapping(diffusion_model.state_dict(), output_onnx)
            output_mapping = os.path.join(os.path.dirname(output_onnx), "weights_mapping.json")
            output_mapping = ensure_temp_or_output_path(output_mapping)
            os.makedirs(os.path.dirname(output_mapping), exist_ok=True)
            with open(output_mapping, "w", encoding="utf-8") as mapping_file:
                json.dump({"weights_name_mapping": weights_name_mapping, "weights_shape_mapping": weights_shape_mapping}, mapping_file, indent=4)
        else:
            output_mapping = None

        required_llmadapter = model_type == SupportedModelType.Anima
        llm_adapter = diffuser.llm_adapter if required_llmadapter else None
        model_management.unload_all_models()
        # GPU dummy inputs would otherwise remain resident during the TRT build.
        del inputs, tracing_model, diffuser, diffusion_model, model_patcher
        gc.collect()
        model_management.soft_empty_cache()

        print("[TensorRT] Building TensorRT Engine ...")

        output_dir = os.path.join(folder_paths.get_output_directory(), os.path.dirname(spec["filename_prefix"]))
        basename_prefix = os.path.basename(spec["filename_prefix"])
        basename = basename_prefix + os.path.splitext(os.path.basename(model_name))[0] + "_" + _trt_spec_to_string(spec) + f"_{precision}.{model_type.name}"
        trt_output_path = os.path.join(output_dir, f"{basename}.engine")
        os.makedirs(os.path.dirname(trt_output_path), exist_ok=True)

        _build_tensorrt_engine(
            output_onnx, input_names, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, trt_output_path, enable_lora=spec["enable_lora"], workspace_gib=spec.get("workspace_gib", 1)
        )
        with open(trt_output_path + ".json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        print(f"[TensorRT] Conversion completed. Saved to {trt_output_path}")

        if not (enable_lora or required_llmadapter):
            return io.NodeOutput()

        # Need to merge files into a bundle for easier distribution
        bundle_out = os.path.join(output_dir, f"{basename}.bundle")
        if required_llmadapter:
            print("[TensorRT] Exporting LLMAdapter to ONNX ...")
            output_onnx_llm_adapter = os.path.normpath(os.path.join(temp_dir, f"trt_{time.time()}", "llm_adapter.onnx"))
            _export_anima_llmadapter_to_onnx(llm_adapter, output_onnx_llm_adapter, dtype)
            print(f"[TensorRT] LLMAdapter ONNX export completed. Saved to {output_onnx_llm_adapter}")
            bundle = ModelBundle.from_onnx_and_trt_engine(
                onnx_path=output_onnx_llm_adapter,
                trt_engine_path=trt_output_path,
                output_path=bundle_out,
                replace_source=True,
            )
        else:
            bundle = ModelBundle.from_trt_engine(
                trt_engine_path=trt_output_path,
                output_path=bundle_out,
                replace_source=True
            )

        if enable_lora and output_mapping:
            with open(output_mapping, encoding="utf-8") as mapping_file:
                json_mapping = json.load(mapping_file)
            bundle.save_weights_mapping(json_mapping["weights_name_mapping"], json_mapping["weights_shape_mapping"])
            bundle.metadata = {"source_model": model_source, "export_metadata": metadata}
        else:
            bundle.metadata = {"export_metadata": metadata}
        bundle.close()
        
        return io.NodeOutput()


def _get_model_name_options() -> list[str]:
    checkpoint_names = folder_paths.get_filename_list("checkpoints")
    diffusion_names = folder_paths.get_filename_list("diffusion_models")
    seen: set[str] = set()
    options: list[str] = []
    model_suffixes = [".safetensors", ".ckpt"]
    for name in checkpoint_names + diffusion_names:
        if not any(name.endswith(suffix) for suffix in model_suffixes):
            continue
        if name not in seen:
            options.append(name)
            seen.add(name)
    return options

def _validate_export_env():
    if hasattr(comfy.cli_args, 'enables_dynamic_vram') and comfy.cli_args.enables_dynamic_vram():
        red_start = "\033[91m"
        reset = "\033[0m"
        
        error_msg = (
            "\n[Compatibility Error] ONNX Export cannot run while Dynamic VRAM is enabled.\n"
            "The memory management conflicts with the PyTorch FX graph decomposition.\n"
            f"{red_start}Please restart ComfyUI with the '--disable-dynamic-vram' command line argument to proceed.\n"
            f"起動引数 '--disable-dynamic-vram' を付けてComfyUIを再起動してください{reset}"
        )
        raise RuntimeError(error_msg)

@contextmanager
def _prepare_model_for_onnx_export():
    """
    Temporarily replace custom comfy_kitchen RoPE split-half ops with equivalent
    eager PyTorch implementations so torch.onnx.export(dynamo=True) can lower them.
    Also enables the generic training-mode fallbacks used by some ComfyUI models.
    """
    patched_attrs: list[tuple[str, Any]] = []
    prev_in_training = comfy.model_management.in_training
    comfy.model_management.in_training = True

    ck = getattr(comfy.quant_ops, "ck", None)
    attention = comfy.ldm.modules.attention
    originals = [(attention, "optimized_attention", attention.optimized_attention),
                 (attention, "optimized_attention_masked", attention.optimized_attention_masked),
                 (comfy.ops, "scaled_dot_product_attention", comfy.ops.scaled_dot_product_attention)]
    attention.optimized_attention = attention.attention_pytorch
    attention.optimized_attention_masked = attention.attention_pytorch
    comfy.ops.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    def _apply_rope_split_half1_eager(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
        t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
        return t_out.movedim(-1, -2).reshape(*x.shape).type_as(x)

    if ck is not None:
        if hasattr(ck, "apply_rope_split_half1"):
            patched_attrs.append(("apply_rope_split_half1", ck.apply_rope_split_half1))
            ck.apply_rope_split_half1 = _apply_rope_split_half1_eager

        if hasattr(ck, "apply_rope_split_half"):
            patched_attrs.append(("apply_rope_split_half", ck.apply_rope_split_half))
            ck.apply_rope_split_half = lambda xq, xk, freqs_cis: (
                _apply_rope_split_half1_eager(xq, freqs_cis),
                _apply_rope_split_half1_eager(xk, freqs_cis),
            )

    try:
        # Keep fused CUDA/Sage/xFormers kernels out of the ONNX graph.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        comfy.model_management.in_training = prev_in_training
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        if ck is not None:
            for attr_name, original_value in reversed(patched_attrs):
                setattr(ck, attr_name, original_value)
    
def _load_model_from_basename(model_name: str, dtype: torch.dtype = torch.float16) -> tuple[str, comfy.model_patcher.ModelPatcher]:
    model_source = None
    model_patcher = None
    
    checkpoint_path = folder_paths.get_full_path("checkpoints", model_name)
    diffusion_path = folder_paths.get_full_path("diffusion_models", model_name)

    has_checkpoint = checkpoint_path is not None
    has_diffusion = diffusion_path is not None
    
    if has_checkpoint and has_diffusion:
        raise ValueError(
            f"Model name exists in both checkpoints and diffusion_models: {model_name}. "
            "Please rename one of them."
        )
    if has_checkpoint:
        model_source = checkpoint_path
        model_patcher, *_ = sd.load_checkpoint_guess_config( # pyright: ignore[reportUnknownMemberType]
            checkpoint_path,
            output_vae=False,
            output_clip=False,
            output_clipvision=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options={"dtype": dtype}, disable_dynamic=True
        )
        if model_patcher is None:
            raise ValueError(f"Failed to load model patcher from checkpoint: {checkpoint_path}")
    elif has_diffusion:
        model_source = diffusion_path
        model_patcher = sd.load_diffusion_model(diffusion_path, model_options={"dtype": dtype}, disable_dynamic=True)
    else:
        raise FileNotFoundError(f"Model not found in checkpoints or diffusion_models: {model_name}")
    
    return model_source, model_patcher

def _adjust_spec_range(sepc: TRTSpec) -> TRTSpec:
        min_batch_size, opt_batch_size, max_batch_size = sepc["min_batch_size"], sepc["opt_batch_size"], sepc["max_batch_size"]
        min_height, opt_height, max_height = sepc["min_height"], sepc["opt_height"], sepc["max_height"]
        min_width, opt_width, max_width = sepc["min_width"], sepc["opt_width"], sepc["max_width"]
        min_context_mult, opt_context_mult, max_context_mult = sepc["min_context_mult"], sepc["opt_context_mult"], sepc["max_context_mult"]

        def _adjust_range(min_val: int, max_val: int, opt_val: int) -> tuple[int, int]:
            new_min = opt_val if min_val == 0 else min(min_val, opt_val)
            new_max = opt_val if max_val == 0 else max(max_val, opt_val)
            return new_min, new_max

        min_batch_size, max_batch_size = _adjust_range(min_batch_size, max_batch_size, opt_batch_size)
        min_height, max_height = _adjust_range(min_height, max_height, opt_height)
        min_width, max_width = _adjust_range(min_width, max_width, opt_width)
        min_context_mult, max_context_mult = _adjust_range(min_context_mult, max_context_mult, opt_context_mult)

        sepc["min_batch_size"], sepc["max_batch_size"] = min_batch_size, max_batch_size
        sepc["min_height"], sepc["max_height"] = min_height, max_height
        sepc["min_width"], sepc["max_width"] = min_width, max_width
        sepc["min_context_mult"], sepc["max_context_mult"] = min_context_mult, max_context_mult
        return sepc

def _get_context_features(model_type: SupportedModelType, unet_config: Optional[dict[str, Any]] = None) -> tuple[int, int, int, torch.dtype]:
    match model_type:
        case SupportedModelType.SD3:
            if unet_config:
                context_embedder_config = unet_config.get("context_embedder_config", None)
                if context_embedder_config:
                    context_dim = context_embedder_config.get("params", {}).get("in_features", None)
                    context_len = 154
                    context_len_min = 77
                else:
                    _context_dim = unet_config.get("context_dim", None)
                    if _context_dim is None:
                        raise NotImplementedError("Unsupported model configuration: context_embedder_config or context_dim is required for SD3")
                    context_dim = _context_dim
                    context_len = 77
                    context_len_min = 77
                dtype = unet_config.get("dtype", torch.float32)
            else:
                raise NotImplementedError("Unsupported model configuration: context_embedder_config is required for SD3 if unet_config is not provided")
            
        case SupportedModelType.AuraFlow:
            context_dim = 2048
            context_len = 512
            context_len_min = 512
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.Flux:
            _context_dim = unet_config.get("context_in_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_in_dim is required for Flux")
            context_dim = _context_dim
            context_len = 512
            context_len_min = 256
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.Anima:
            _context_dim = unet_config.get("crossattn_emb_channels", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: crossattn_emb_channels is required for Anima")
            context_dim = _context_dim
            context_len = 512
            context_len_min = 512
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.SVD:
            _context_dim = unet_config.get("context_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_dim is required for SVD/SVD_XT")
            context_dim = _context_dim
            context_len = 1
            context_len_min = 1
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.SD15:
            _context_dim = unet_config.get("context_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_dim is required for SD15")
            context_dim = _context_dim
            context_len = 77
            context_len_min = 77
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32
        case SupportedModelType.SDXL:
            _context_dim = unet_config.get("context_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_dim is required for SDXL")
            context_dim = _context_dim
            context_len = 77
            context_len_min = 77
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32
        case _:
            assert_never(model_type)

    return context_len_min, context_len, context_dim, dtype

def _build_onnx_tracing_input(
    unet_config: Optional[dict[str, Any]], model_type: SupportedModelType,
    bs_min: int, bs_opt: int, bs_max: int,
    ctx_min: int, ctx_opt: int, ctx_max: int,
    min_height: int, opt_height: int, max_height: int,
    min_width: int, opt_width: int, max_width: int,
    context_len_min: int, context_len: int, context_dim: int, dtype: torch.dtype,
    use_rope3d: bool, num_video_frames: int, y_dim: int
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    list[str],
    list[str],
    dict[str, dict[int, Any]],
]:

    batch_size_dim = torch.export.Dim("batch_size", min=bs_min, max=bs_max) if bs_min != bs_max else bs_opt
    height_dim = torch.export.Dim("height", min=min_height // 8, max=max_height // 8) if min_height != max_height else opt_height // 8
    width_dim = torch.export.Dim("width", min=min_width // 8, max=max_width // 8) if min_width != max_width else opt_width // 8
    actual_ctx_min = context_len_min * ctx_min
    actual_ctx_max = context_len * ctx_max
    actual_ctx_opt = context_len * ctx_opt
    num_embeds_dim = torch.export.Dim("num_embeds", min=actual_ctx_min, max=actual_ctx_max) if actual_ctx_min != actual_ctx_max else actual_ctx_opt
    num_video_frames_dim = num_video_frames if use_rope3d else None

    input_names = ["latent", "timestep", "context"]
    output_names = ["output"]
    dynamic_shapes: dict[str, dict[int, Any]] = {
        "latent": {0: batch_size_dim, 2: height_dim, 3: width_dim},
        "timestep": {0: batch_size_dim},
        "context": {0: batch_size_dim, 1: num_embeds_dim}
    }

    # Build shapes and dummy inputs
    input_channels: int = int(unet_config.get("in_channels", 4)) if unet_config is not None else 4
    
    inputs_shapes_min: tuple[tuple[int, ...], ...] = (
        (bs_min, input_channels, min_height // 8, min_width // 8),
        (bs_min,),
        (bs_min, context_len_min * ctx_min, context_dim),
    )
    inputs_shapes_opt: tuple[tuple[int, ...], ...] = (
        (bs_opt, input_channels, opt_height // 8, opt_width // 8),
        (bs_opt,),
        (bs_opt, context_len * ctx_opt, context_dim),
    )
    inputs_shapes_max: tuple[tuple[int, ...], ...] = (
        (bs_max, input_channels, max_height // 8, max_width // 8),
        (bs_max,),
        (bs_max, context_len * ctx_max, context_dim),
    )

    # Anima latent: (bs, input_channels, num_video_frames, h//8, w//8)
    if use_rope3d and num_video_frames >= 1:
        inputs_shapes_min = ((bs_min, input_channels, num_video_frames, min_height // 8, min_width // 8),) + inputs_shapes_min[1:]
        inputs_shapes_opt = ((bs_opt, input_channels, num_video_frames, opt_height // 8, opt_width // 8),) + inputs_shapes_opt[1:]
        inputs_shapes_max = ((bs_max, input_channels, num_video_frames, max_height // 8, max_width // 8),) + inputs_shapes_max[1:]
        dynamic_shapes["latent"] = {0: batch_size_dim, 2: num_video_frames_dim, 3: height_dim, 4: width_dim}

    # SDXL
    if y_dim > 0:
        input_names.append("vector_cond")
        dynamic_shapes["vector_cond"] = {0: batch_size_dim}
        inputs_shapes_min += ((bs_min, y_dim),)
        inputs_shapes_opt += ((bs_opt, y_dim),)
        inputs_shapes_max += ((bs_max, y_dim),)

    # FLUX
    extra_input: dict[str, tuple[int, ...]] = {}
    if model_type == SupportedModelType.Flux:
        extra_input = {"guidance": ()}

    for k, v in extra_input.items():
        input_names.append(k)
        dynamic_shapes[k] = {0: batch_size_dim}
        inputs_shapes_min += ((bs_min,) + v,)
        inputs_shapes_opt += ((bs_opt,) + v,)
        inputs_shapes_max += ((bs_max,) + v,)

    device = cast(torch.device, model_management.get_torch_device())
    inputs: tuple[torch.Tensor, ...] = tuple(
        torch.zeros(shape, device=device, dtype=torch.float32 if name == "timestep" else dtype)
        for name, shape in zip(input_names, inputs_shapes_opt))
    
    return inputs, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, input_names, output_names, dynamic_shapes

def _build_onnx_tracing_model(is_svd: bool, diffuser: nn.Module, transformer_options: dict[str, Any], num_video_frames: Optional[int]) -> nn.Module:
    
    if is_svd:
        if num_video_frames is None:
            raise ValueError("num_video_frames is required for SVD models")
        class SVDTracingModel(nn.Module):
            @override
            def __init__(self, diffuser: nn.Module, num_video_frames: int, transformer_options: dict[str, Any]):
                super().__init__()
                self.diffuser = diffuser
                self.num_video_frames = num_video_frames
                self.transformer_options = transformer_options

            def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, vector_cond: torch.Tensor):
                return self.diffuser(latent, timestep, context, vector_cond, num_video_frames=self.num_video_frames, transformer_options=self.transformer_options)

        return SVDTracingModel(diffuser, num_video_frames, transformer_options).eval()

    else:
        class TracingModel(nn.Module):
            @override
            def __init__(self, diffuser: nn.Module, transformer_options: dict[str, Any]):
                super().__init__()
                self.diffuser = diffuser
                self.transformer_options = transformer_options

            def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, vector_cond: Optional[torch.Tensor] = None, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
                # """kwargs is not supported in torch.onnx.export""" with dynamo=True, so we need to handle optional inputs manually
                if vector_cond is not None and guidance is not None:
                    return self.diffuser(latent, timestep, context, vector_cond, guidance, transformer_options=self.transformer_options)
                elif vector_cond is not None:
                    return self.diffuser(latent, timestep, context, vector_cond, transformer_options=self.transformer_options)
                elif guidance is not None:
                    return self.diffuser(latent, timestep, context, guidance=guidance, transformer_options=self.transformer_options)
                else:
                    return self.diffuser(latent, timestep, context, transformer_options=self.transformer_options)

        return TracingModel(diffuser, transformer_options).eval()

def _build_tensorrt_engine(
    target_onnx: str,
    input_names: list[str],
    inputs_shapes_min: tuple[tuple[int, ...], ...],
    inputs_shapes_opt: tuple[tuple[int, ...], ...],
    inputs_shapes_max: tuple[tuple[int, ...], ...],
    output_path: str,
    enable_lora: bool,
    workspace_gib: int = 1,
) -> None:
    if trt is None:
        raise RuntimeError("TensorRT is not installed but required for compilation.")

    builder = trt.Builder(trt_logger)

    network = builder.create_network(network_creation_flags())
    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(target_onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx)) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    if not success:
        raise RuntimeError("ONNX parse ERROR")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))
    if enable_lora:
        config.set_flag(trt.BuilderFlag.REFIT)
    profile = builder.create_optimization_profile()
    
    temp_dir = folder_paths.get_temp_directory()
    cache_key = hashlib.sha256(f"{trt.__version__}/{torch.cuda.get_device_name()}/{torch.cuda.get_device_capability()}/{workspace_gib}/{enable_lora}".encode()).hexdigest()[:16]
    timing_cache_path = os.path.join(temp_dir, "timing_cache", f"timing_cache_{cache_key}.trt")
    timing_cache_path = ensure_temp_or_output_path(timing_cache_path)
    os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    else:
        buffer = b""
    timing_cache = config.create_timing_cache(buffer)
    if not config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError("Incompatible TensorRT timing cache; remove this node's timing cache and retry.")
    
    for k in range(len(input_names)):
        # Python set_shape raises ValueError and returns None on success.
        profile.set_shape(input_names[k], inputs_shapes_min[k], inputs_shapes_opt[k], inputs_shapes_max[k])
        
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the optimization profile.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed. Check the TensorRT log above (types, unsupported operators, or VRAM). ONNX files were retained.")
    
    output_path = ensure_temp_or_output_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    timing_cache_path = ensure_temp_or_output_path(timing_cache_path)
    os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(config.get_timing_cache().serialize()))

def _get_weights_mapping(state_dict: dict[str, torch.Tensor], onnx_model_path: str) -> tuple[dict[str, str], dict[str, tuple[tuple[int, ...], bool]]]:

    print("Loading ONNX model to extract initializers...")
    model = onnx.load(onnx_model_path) # pyright: ignore[reportUnknownMemberType]
    
    initializer_hash_mapping: dict[str, tuple[int, tuple[int, ...]]] = {}
    
    print("Computing hashes for ONNX initializers...")
    for initializer in tqdm.tqdm(model.graph.initializer, desc="ONNX Initializers"):
        initializer_data = onnx.numpy_helper.to_array(initializer).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = (initializer_hash, tuple(initializer_data.shape))

    weights_name_mapping: WeightsNameMap = {}
    weights_shape_mapping: WeightsShapeMap = {}
    initializers_mapped: set[str] = set()

    print("Computing hashes for PyTorch weights and finding matches...")
    for wt_name, wt in tqdm.tqdm(state_dict.items(), desc="PyTorch Weights"):
        wt_np = wt.to(torch.float32).cpu().detach().numpy().astype(np.float16)
        wt_hash = hash(wt_np.data.tobytes())
        wt_t_hash = hash(np.transpose(wt_np).data.tobytes())
        
        for initializer_name, (initializer_hash, initializer_shape) in initializer_hash_mapping.items():
            if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                assert initializer_name not in initializers_mapped, f"Duplicate mapping for {initializer_name}"
                
                weights_name_mapping[wt_name] = initializer_name
                initializers_mapped.add(initializer_name)
                
                is_transpose = False if wt_hash == initializer_hash else True
                weights_shape_mapping[wt_name] = (initializer_shape, is_transpose)
                break
                
    print(f"Mapped {len(weights_name_mapping)} / {len(state_dict)} weights.")
        
    return weights_name_mapping, weights_shape_mapping


def _trt_spec_to_string(spec: TRTSpec) -> str:
    s = ""
    if spec["min_batch_size"] == spec["max_batch_size"]:
        s += f"BS{spec['opt_batch_size']}_"
    else:
        s += f"BS{spec['min_batch_size']}-{spec['opt_batch_size']}-{spec['max_batch_size']}_"
    
    if spec["min_width"] == spec["max_width"]:
        s += f"W{spec['opt_width']}_"
    else:        s += f"W{spec['min_width']}-{spec['opt_width']}-{spec['max_width']}_"

    if spec["min_height"] == spec["max_height"]:
        s += f"H{spec['opt_height']}_"
    else:
        s += f"H{spec['min_height']}-{spec['opt_height']}-{spec['max_height']}_"
    
    if spec["min_context_mult"] == spec["max_context_mult"]:
        s += f"CM{spec['opt_context_mult']}"
    else:
        s += f"CM{spec['min_context_mult']}-{spec['opt_context_mult']}-{spec['max_context_mult']}"

    if spec["num_video_frames"] > 1:
        s += f"_VF{spec['num_video_frames']}"

    if spec["enable_lora"]:
        s += "_LoRA"

    return s

def _export_anima_llmadapter_to_onnx(llm_adapter: "LLMAdapter", output_path: str, dtype: torch.dtype):
    output_path = ensure_temp_or_output_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device("cpu") # cast(torch.device, model_management.get_torch_device())
    
    tracing_model = llm_adapter.to(device)
    if device.type == "cpu":
        dtype = torch.float32
        tracing_model = llm_adapter.to(device)
        for _, param in tracing_model.named_parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.float()
    tracing_model.eval()

    input_names = ["source_hidden_states", "target_input_ids", "target_attention_mask", "source_attention_mask"]
    output_names = ["output"]
    batch_size = 1
    source_seq_len = torch.export.Dim("source_seq_len", min=1, max=512) # Anima's LLMAdapter supports variable sequence length up to 512
    target_seq_len = torch.export.Dim("target_seq_len", min=1, max=512) # Anima's LLMAdapter supports variable sequence length up to 512 for target (prompt) inputs

    dynamic_shapes: dict[str, dict[int, torch.export.Dim | int]] = {
        "source_hidden_states": {0: batch_size, 1: source_seq_len},
        "target_input_ids": {0: batch_size, 1: target_seq_len},
        "target_attention_mask": {0: batch_size, 2: target_seq_len, 3: target_seq_len},
        "source_attention_mask": {0: batch_size, 2: target_seq_len, 3: source_seq_len},
    }

    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
        torch.randn(batch_size, source_seq_len.max, 1024, device=device, dtype=dtype), # source_hidden_states
        torch.randint(0, 32128, (batch_size, target_seq_len.max), device=device),      # target_input_ids (Long型)
        torch.ones(batch_size, 1, target_seq_len.max, target_seq_len.max, device=device, dtype=dtype),        # target_attention_mask
        torch.ones(batch_size, 1, target_seq_len.max, source_seq_len.max, device=device, dtype=dtype),    # source_attention_mask
    )
    # Anima LLMAdapter is supported in opset 25
    
    torch.onnx.export( # pyright: ignore[reportUnknownMemberType]
        tracing_model,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=25,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
    )
