"""Small, testable helpers shared by the exporter and loader."""
import json
import os
import platform

import torch
import tensorrt as trt


def network_creation_flags():
    # TRT 11 always uses strong typing. The enum still exists in some wheels,
    # but it is no longer necessary (and must not be assumed to exist).
    if int(trt.__version__.split('.')[0]) >= 11:
        return 0
    return 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)


def legacy_dynamic_axes(names, minimum, maximum, output_names):
    axes = {}
    for name, low, high in zip(names, minimum, maximum):
        dynamic = {}
        for axis, (a, b) in enumerate(zip(low, high)):
            if a != b:
                if axis == 0:
                    symbol = 'batch_size'
                elif name == 'latent' and axis in (2, 3):
                    symbol = {2: 'height', 3: 'width'}[axis]
                else:
                    symbol = f'{name}_dim{axis}'
                dynamic[axis] = symbol
        if dynamic:
            axes[name] = dynamic
    if 'latent' in axes:
        for name in output_names:
            axes[name] = axes['latent'].copy()
    return axes


def split_batch_size(batch, minimum, maximum):
    for size in range(min(batch, maximum), minimum - 1, -1):
        if batch % size == 0:
            return size
    raise ValueError(f'Batch {batch} cannot be split into engine batches [{minimum}, {maximum}].')


def make_export_metadata(model, model_type, spec, input_names, minimum, optimum, maximum):
    # Store the configuration chosen by ComfyUI, not a guess based on a filename.
    unet_config = {k: v for k, v in model.model_config.unet_config.items()
                   if not isinstance(v, torch.dtype)}
    metadata = {
        'schema_version': 1,
        'model_family': model_type.name,
        'prediction_type': model.model_type.name,
        'sampling_settings': model.model_config.sampling_settings,
        'unet_config': unet_config,
        'precision': spec.get('precision', 'fp16'),
        'profiles': {name: {'min': low, 'opt': opt, 'max': high}
                     for name, low, opt, high in zip(input_names, minimum, optimum, maximum)},
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'tensorrt': trt.__version__, 'platform': platform.system(),
        'gpu': torch.cuda.get_device_name(),
        'compute_capability': list(torch.cuda.get_device_capability()),
    }
    # Fail before a costly build if a future ComfyUI configuration isn't JSON.
    return json.loads(json.dumps(metadata))


def read_engine_metadata(path):
    metadata_path = path + '.json'
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, encoding='utf-8') as source:
        metadata = json.load(source)
    if metadata.get('schema_version') != 1:
        raise ValueError(f'Unsupported engine metadata schema: {metadata_path}')
    return metadata


def sdxl_sampling_config(metadata, override):
    if override not in ('auto', 'eps', 'v_prediction', 'v_prediction_zsnr'):
        raise ValueError(f'Unknown sampling override: {override}')
    if override == 'auto':
        if not metadata:
            raise ValueError('SDXL auto sampling needs the .engine.json sidecar. '
                             'Re-export, or explicitly choose eps/v_prediction/v_prediction_zsnr.')
        prediction = metadata['prediction_type']
        settings = dict(metadata['sampling_settings'])
    else:
        prediction = 'EPS' if override == 'eps' else 'V_PREDICTION'
        settings = dict((metadata or {}).get('sampling_settings', {}))
        settings['zsnr'] = override == 'v_prediction_zsnr'
    if prediction not in ('EPS', 'V_PREDICTION', 'V_PREDICTION_EDM'):
        raise ValueError(f'This SDXL patch does not support sampling type {prediction}.')
    return prediction, settings
