from typing import Any, Optional, cast, Literal
import threading

import numpy as np
from tqdm import tqdm

import torch
import tensorrt as trt

import comfy.lora
import comfy.model_management

from ..definitions import SupportedModelType, WeightsNameMap, WeightsShapeMap, PatchType, ModelInputNames, ModelInputMapping, trt_logger
from ..trt_utils import trt_datatype_to_torch, torch_dtype_to_trt
from ..trt_compat import split_batch_size

SupportedModelName = [e.name for e in SupportedModelType]

class TRTDiffuser:
    def __init__(self, engine: trt.ICudaEngine, weight_map: Optional[WeightsNameMap] = None, shape_map: Optional[WeightsShapeMap] = None) -> None:        
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize this engine. Rebuild it with this GPU/TensorRT version; see the TensorRT log.")
        self.engine = engine
        
        trt_dtype = self.engine.get_tensor_dtype(self.engine.get_tensor_name(0))

        self.dtype = trt_datatype_to_torch(trt_dtype)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context (check free VRAM).")
        self.device = comfy.model_management.get_torch_device()
        self._execution_lock = threading.Lock()
        self._last_execution = None

        self.model_input_names: ModelInputNames = {
            "latent": "",
            "timestep": "",
            "context": None,
            "vector_cond": None
        }
        self.input_aliases_map: ModelInputMapping = {
            "latent": ["latent", "x", "cx"],
            "timestep": ["timestep", "timesteps", "t"],
            "context": ["context", "ctx"],
            "vector_cond": ["vector_cond", "y"]
        }
        self.weight_mapping = weight_map
        self.shape_mapping = shape_map

        self.model_output_names: list[str] = []
        self.patches: dict[str, list[PatchType]] = {}
        self.source_state_dict: dict[str, torch.Tensor] = {}
        self._last_refit_layers: set[str] = set()
        self.refit_compute_device: torch.device = cast(torch.device, comfy.model_management.get_torch_device())

        self._rename_inputs()
        latent_name = self.model_input_names["latent"]
        if not latent_name or not self.model_input_names["timestep"]:
            raise ValueError("Engine is missing latent/timestep inputs.")
        self.dtype = trt_datatype_to_torch(self.engine.get_tensor_dtype(latent_name))

    def _rename_inputs(self):
        engine_model_input_names: list[str] = []
        engine_model_output_names: list[str] = []

        # tensorrt is 8.5+ by requirement
        num_io_tensors = self.engine.num_io_tensors
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                engine_model_input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                engine_model_output_names.append(name)

        for key, aliases in self.input_aliases_map.items():
            for alias in cast(list[str], aliases):
                if alias in engine_model_input_names:
                    self.model_input_names[key] = alias
                    break

        self.model_output_names = engine_model_output_names

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("control") is not None:
            raise ValueError("ControlNet is not part of this TensorRT graph. Use the PyTorch model for this workflow.")
        options = kwargs.get("transformer_options") or {}
        if any(options.get(key) for key in ("patches", "patches_replace", "wrappers")):
            raise ValueError("Runtime transformer patches cannot be applied to a compiled TensorRT graph.")
        if latent.device.type != "cuda" or latent.device != self.device:
            raise ValueError(f"Engine is on {self.device}; received latent on {latent.device}.")
        model_inputs: dict[str, torch.Tensor] = {self.model_input_names["latent"]: latent, self.model_input_names["timestep"]: timestep}

        for arg, value in kwargs.items():
            for key, aliases in self.input_aliases_map.items():
                if arg in cast(list[str], aliases) and self.model_input_names[key] is not None and value is not None:
                    model_inputs[self.model_input_names[key]] = value
                    break
        
        expected = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                    if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT}
        if expected != set(model_inputs):
            raise ValueError(f"Engine input mismatch: expected {sorted(expected)}, received {sorted(model_inputs)}")
        if len(self.model_output_names) != 1:
            raise ValueError("Only single-output diffusion engines are supported.")
        for name in expected | set(self.model_output_names):
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Host tensor binding is unsupported: {name}")
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Non-linear tensor binding is unsupported: {name}")

        batch = latent.shape[0]
        low, _, high = self.engine.get_tensor_profile_shape(self.model_input_names["latent"], 0)
        chunk_size = split_batch_size(batch, low[0], high[0])
        output_name = self.model_output_names[0]
        # An execution context cannot be reconfigured while its previous enqueue
        # is still running, including calls made from a different CUDA stream.
        with self._execution_lock, torch.cuda.device(latent.device):
            if self._last_execution is not None:
                self._last_execution.synchronize()
            stream = torch.cuda.current_stream(latent.device)
            converted = {}
            for name, tensor in model_inputs.items():
                if tensor.ndim == 0:
                    tensor = tensor.reshape(1)
                if tensor.shape[0] == 1 and batch != 1:
                    tensor = tensor.expand(batch, *tensor.shape[1:])
                if tensor.shape[0] != batch:
                    raise ValueError(f"{name} batch {tensor.shape[0]} does not match latent batch {batch}.")
                tensor = tensor.to(device=latent.device, dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(name))).contiguous()
                shape = (chunk_size, *tensor.shape[1:])
                minimum, _, maximum = self.engine.get_tensor_profile_shape(name, 0)
                if len(shape) != len(minimum) or any(v < a or v > b for v, a, b in zip(shape, minimum, maximum)):
                    raise ValueError(f"{name} shape {shape} is outside engine profile {tuple(minimum)}..{tuple(maximum)}. Re-export for this resolution/prompt length.")
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f"TensorRT rejected input shape {name}={shape}")
                converted[name] = tensor
            missing = self.context.infer_shapes()
            if missing:
                raise RuntimeError(f"TensorRT shapes remain unspecified: {missing}")
            shape = tuple(self.context.get_tensor_shape(output_name))
            if not shape or any(v <= 0 for v in shape) or shape[0] != chunk_size:
                raise RuntimeError(f"Invalid TensorRT output shape: {shape}")
            output = torch.empty((batch, *shape[1:]), device=latent.device,
                                 dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_name)))
            try:
                for start in range(0, batch, chunk_size):
                    bindings = {name: tensor[start:start + chunk_size] for name, tensor in converted.items()}
                    bindings[output_name] = output[start:start + chunk_size]
                    for name, tensor in bindings.items():
                        if not self.context.set_tensor_address(name, tensor.data_ptr()):
                            raise RuntimeError(f"TensorRT rejected tensor address for {name}")
                        tensor.record_stream(stream)
                    if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                        raise RuntimeError("TensorRT execute_async_v3 failed; see the TensorRT log.")
            finally:
                # Keep the allocator and the next caller aware of outstanding work,
                # even when a later chunk fails after an earlier successful enqueue.
                self._last_execution = torch.cuda.Event()
                self._last_execution.record(stream)
            return output

    def refit(self):
        if int(trt.__version__.split('.')[0]) >= 11:
            raise RuntimeError("LoRA/Refit is outside the validated TensorRT 11 SDXL path. Use an engine exported with enable_lora=False.")
        self._validate_refit_support()
        refitter = trt.Refitter(self.engine, trt_logger)
        keep_weights: dict[str, torch.Tensor] = {}

        target_layers = list(self._last_refit_layers)
        for layer in self.patches:
            if self._should_refit_layer(layer) and layer not in target_layers:
                target_layers.append(layer)

        pbar = tqdm(target_layers, desc="[TRT] Calculating and Refitting LoRA Layers", total=len(target_layers))
        weight_device_map = self._schedule_device(target_layers)
        cpu_buffers = self._create_cpu_buffer([layer for layer in target_layers if weight_device_map[layer] == "cpu"])
        event = torch.cuda.Event()
        event.record()
        try:
            for layer in pbar:
                pbar.set_postfix({"layer": layer.split('.')[-1]}) # pyright: ignore[reportUnknownMemberType]

                base_weight = self._build_base_weight(layer)
                patch_list = self.patches.get(layer, [])
                if patch_list:
                    base_weight = self._apply_patches(layer, base_weight, patch_list)
                final_weight = self._prepare_final_weight(layer, base_weight)
                if weight_device_map[layer] == "cpu":
                    cpu_buffers[layer].copy_(final_weight, non_blocking=True)
                    keep_weights[layer] = cpu_buffers[layer]
                else:
                    host_weight = self._set_refit_weights(refitter, layer, final_weight)
                    keep_weights[layer] = host_weight
        finally:
            pbar.close()
            event.record()
            event.synchronize()
            cpu_layers = [layer for layer in target_layers if weight_device_map[layer] == "cpu"]
            for cpu_layer in cpu_layers:
                host_weight = self._set_refit_weights(refitter, cpu_layer, cpu_buffers[cpu_layer])
        
        self._commit_refit(refitter, keep_weights)
        self._last_refit_layers = {layer for layer in target_layers if self._should_refit_layer(layer)}
        self._finalize_refit(keep_weights)

    def _validate_refit_support(self) -> None:
        if self.patches and not (self.weight_mapping and self.shape_mapping):
            raise RuntimeError("This model does not support LoRA.")
        if (self.patches or self._last_refit_layers) and not self.source_state_dict:
            raise RuntimeError("Source state_dict is required for TensorRT refit but not found.")

    def _should_refit_layer(self, layer: str) -> bool:
        assert self.weight_mapping is not None, "Weight mapping is required for refitting but not found."
        return layer in self.weight_mapping

    def _schedule_device(self, layers: list[str]) -> dict[str, Literal["cpu", "cuda"]]:
        assert self.shape_mapping is not None, "Shape mapping is required for refitting but not found."
        free, _ = torch.cuda.mem_get_info("cuda")
        gpu_memory_margin = 1.8 # Alocate 1.8x the weight size on GPU to account for temporary tensors during refit
        weight_device_map: dict[str, Literal["cpu", "cuda"]] = {}
        # Process layers in reverse to prioritize weights scheduled for CPU by assigning them first.
        # This allows asynchronous transfers to start earlier, maximizing overlap with GPU tasks.
        for layer in reversed(layers):
            shapes = list(self.shape_mapping[layer][0])
            weight_size = self.dtype.itemsize * np.prod(shapes) * gpu_memory_margin
            free -= weight_size
            if free > 0:
                weight_device_map[layer] = "cuda"
            else:
                weight_device_map[layer] = "cpu"
        return weight_device_map
    
    def _create_cpu_buffer(self, layers: list[str]) -> dict[str, torch.Tensor]:
        assert self.shape_mapping is not None, "Shape mapping is required for refitting but not found."
        alignment = 256
        pointer = 0
        layer_range: dict[str, tuple[int, int]] = {}
        for layer in layers:
            shapes = list(self.shape_mapping[layer][0])
            weight_size = int(self.dtype.itemsize * np.prod(shapes))
            layer_range[layer] = (pointer, pointer + weight_size)
            pointer += (weight_size + alignment - 1) // alignment * alignment

        buffer_size = pointer
        storage = torch.UntypedStorage(buffer_size, device="cpu").pin_memory()
        cpu_buffers: dict[str, torch.Tensor] = {}
        for layer in layers:
            start, end = layer_range[layer]
            buffer = cast(torch.UntypedStorage, storage[start:end])
            cpu_buffers[layer] = torch.tensor(buffer, dtype=self.dtype, device="cpu").view(self.shape_mapping[layer][0])

        return cpu_buffers

    def _build_base_weight(self, layer: str) -> torch.Tensor:
        if layer not in self.source_state_dict:
            raise RuntimeError(f"Source weight for layer {layer} not found.")
        return self.source_state_dict[layer].detach().clone().to(device=self.refit_compute_device, dtype=torch.float32)

    def _apply_patches(self, layer: str, base_weight: torch.Tensor, patch_list: list[PatchType]) -> torch.Tensor:
        def _identity_convert(weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return weight

        original_weights = {
            layer: [(base_weight.detach().clone(), _identity_convert)],
        }
        out_weight: torch.Tensor = cast(torch.Tensor, comfy.lora.calculate_weight( # pyright: ignore[reportUnknownMemberType]
            patch_list,
            base_weight,
            layer,
            intermediate_dtype=torch.float32,
            original_weights=original_weights,
        ))
        return out_weight

    def _prepare_final_weight(self, layer: str, base_weight: torch.Tensor) -> torch.Tensor:
        if self.shape_mapping is None or layer not in self.shape_mapping:
            raise RuntimeError(f"Shape mapping is required for refitting layer {layer} but not found.")
        is_transpose = self.shape_mapping[layer][1]
        if is_transpose:
            final_weight = base_weight.T.contiguous()
        else:
            final_weight = base_weight.contiguous()

        # Match engine dtype (FP16/FP32/BF16 depending on engine build).
        return final_weight.to(self.dtype)

    def _set_refit_weights(self, refitter: trt.Refitter, layer: str, final_weight: torch.Tensor) -> torch.Tensor:
        assert self.weight_mapping is not None, "Weight mapping is required for refitting but not found."
        trt_weights = trt.Weights(
            torch_dtype_to_trt(final_weight.dtype),
            final_weight.data_ptr(),
            final_weight.numel(),
        )

        success = refitter.set_named_weights(self.weight_mapping[layer], trt_weights)
        if not success:
            raise RuntimeError(f"Failed to set weights for {layer}")
        return final_weight

    def _commit_refit(self, refitter: trt.Refitter, keep_weights: dict[str, torch.Tensor], num_tries: int = 5, current_try: int = 0) -> None:
        assert self.weight_mapping is not None, "Weight mapping is required for refitting but not found."
        print("[TRT] Committing refit to engine...")
        success = refitter.refit_cuda_engine()
        if success:
            print("[TRT] Engine refit successful.")
        else:
            missing_onnx_weights = refitter.get_missing_weights()
            if current_try >= num_tries or not missing_onnx_weights:
                raise RuntimeError("Final engine refit failed without reporting missing weights.")

            missing_layers = [layer for layer, weight_name in self.weight_mapping.items() if weight_name in missing_onnx_weights]
            
            for layer in missing_layers:
                if layer not in self.source_state_dict:
                    raise RuntimeError(f"Final engine refit failed: Missing weight '{layer}' not found in original state dict.")
                
                print(f"[TRT] WARNING: Missing weight for '{layer}'. Assigning original weight from state dict.")
                base_weight = self._build_base_weight(layer)
                final_weight = self._prepare_final_weight(layer, base_weight)
                host_weight = self._set_refit_weights(refitter, layer, final_weight)
                keep_weights[layer] = host_weight
            
            self._commit_refit(refitter, keep_weights, num_tries, current_try + 1)

    def _finalize_refit(self, keep_weights: dict[str, torch.Tensor]) -> None:
        self.patches = {}
        del keep_weights

    def set_source_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.source_state_dict = state_dict

    def load_state_dict(self, sd: dict[str, Any], strict: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}
