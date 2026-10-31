"""CPU regression tests; do not import the ComfyUI extension entrypoint."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('trt_compat', Path(__file__).parents[1] / 'trt_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class CompatibilityTests(unittest.TestCase):
    def test_trt11_does_not_access_strongly_typed_enum(self):
        with mock.patch.object(compat.trt, '__version__', '11.4.0.106'), \
             mock.patch.object(compat.trt, 'NetworkDefinitionCreationFlag', object()):
            self.assertEqual(compat.network_creation_flags(), 0)
        with mock.patch.object(compat.trt, '__version__', '10.15.1'), \
             mock.patch.object(compat.trt, 'NetworkDefinitionCreationFlag', SimpleNamespace(STRONGLY_TYPED=3)):
            self.assertEqual(compat.network_creation_flags(), 8)

    def test_cfg_batch_splitting(self):
        self.assertEqual(compat.split_batch_size(2, 1, 1), 1)
        self.assertEqual(compat.split_batch_size(6, 1, 4), 3)
        with self.assertRaises(ValueError):
            compat.split_batch_size(3, 2, 2)

    def test_legacy_axes_share_batch_and_output_spatial_dimensions(self):
        axes = compat.legacy_dynamic_axes(
            ['latent', 'timestep', 'context', 'vector_cond'],
            [(1, 4, 64, 64), (1,), (1, 77, 2048), (1, 2816)],
            [(1, 4, 128, 128), (1,), (1, 231, 2048), (1, 2816)], ['output'])
        self.assertEqual(axes['output'], {2: 'height', 3: 'width'})
        self.assertEqual(axes['context'], {1: 'context_dim1'})
        self.assertNotIn('vector_cond', axes)

    def test_sampling_preserves_noobai_vpred_and_zero_snr(self):
        metadata = {'prediction_type': 'V_PREDICTION', 'sampling_settings': {'zsnr': True}}
        self.assertEqual(compat.sdxl_sampling_config(metadata, 'auto'), ('V_PREDICTION', {'zsnr': True}))
        self.assertEqual(compat.sdxl_sampling_config(metadata, 'eps'), ('EPS', {'zsnr': False}))
        self.assertEqual(compat.sdxl_sampling_config(None, 'v_prediction_zsnr'), ('V_PREDICTION', {'zsnr': True}))
        with self.assertRaises(ValueError):
            compat.sdxl_sampling_config(None, 'auto')


if __name__ == '__main__':
    unittest.main()
