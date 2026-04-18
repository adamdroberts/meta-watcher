from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
import tomllib
import unittest
from unittest import mock

import numpy as np

from meta_watcher.config import ModelConfig
from meta_watcher.core import VideoFrame
from meta_watcher.inference import (
    CudaSam31Provider,
    TransformersObjectDetectionProvider,
    _matches_any_prompt,
    _move_inputs_to_model,
    build_provider,
)


class LinuxInferenceInstallTests(unittest.TestCase):
    def test_base_dependencies_keep_numpy_in_sam3_compatible_range(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        dependencies = data["project"]["dependencies"]

        self.assertIn("numpy>=1.26,<2", dependencies)

    def test_linux_extra_installs_sam3_from_main(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        linux_extra = data["project"]["optional-dependencies"]["linux"]

        self.assertIn("sam3 @ git+https://github.com/facebookresearch/sam3.git@main", linux_extra)
        self.assertIn("einops", linux_extra)
        self.assertIn("pycocotools", linux_extra)

    def test_cuda_provider_reports_linux_extra_when_sam3_is_missing(self) -> None:
        provider = CudaSam31Provider("facebook/sam3.1")
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sam3"):
                raise ImportError("No module named 'sam3'")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as exc:
                provider.warmup()

        message = str(exc.exception)
        self.assertIn("facebookresearch/sam3", message)
        self.assertIn('.[desktop,linux]', message)
        self.assertIn("main branch", message)

    def test_cuda_provider_forces_float32_on_warmup(self) -> None:
        provider = CudaSam31Provider("facebook/sam3.1")

        cast_log: list[dict[str, object]] = []
        train_calls: list[bool] = []
        processor_target: list[object] = []

        class FakeTensor:
            def __init__(self, dtype: str, kind: str = "float") -> None:
                self.dtype = dtype
                self.kind = kind
                self.to_calls: list[dict[str, object]] = []

            def is_floating_point(self) -> bool:
                return self.kind == "float"

            def to(self, **kwargs):
                self.to_calls.append(dict(kwargs))
                new_dtype = kwargs.get("dtype", self.dtype)
                clone = FakeTensor(new_dtype, self.kind)
                clone.to_calls = list(self.to_calls)
                return clone

        float_tensor = FakeTensor("bfloat16", "float")
        complex_tensor = FakeTensor("complex64", "complex")

        class FakeModel:
            def _apply(self, fn):
                cast_log.append({"float_in": float_tensor, "complex_in": complex_tensor})
                cast_log.append({"float_out": fn(float_tensor), "complex_out": fn(complex_tensor)})
                return self

            def train(self, mode: bool) -> "FakeModel":
                train_calls.append(mode)
                return self

        fake_model = FakeModel()

        class FakeProcessor:
            def __init__(self, model) -> None:
                processor_target.append(model)

        fake_builder_module = types.ModuleType("sam3.model_builder")
        fake_builder_module.build_sam3_image_model = lambda model_id=None: fake_model
        fake_processor_module = types.ModuleType("sam3.model.sam3_image_processor")
        fake_processor_module.Sam3Processor = FakeProcessor

        class FakeTorch:
            float32 = "float32"

            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

            @staticmethod
            def device(name: str):
                return types.SimpleNamespace(type=name)

            class Tensor:
                pass

        with mock.patch.dict(
            sys.modules,
            {
                "sam3": types.ModuleType("sam3"),
                "sam3.model": types.ModuleType("sam3.model"),
                "sam3.model_builder": fake_builder_module,
                "sam3.model.sam3_image_processor": fake_processor_module,
                "torch": FakeTorch,
            },
        ):
            provider.warmup()

        # The float tensor should end up as float32; the complex tensor
        # should be moved to the device but keep its dtype.
        float_out = cast_log[1]["float_out"]
        complex_out = cast_log[1]["complex_out"]
        self.assertEqual(float_out.dtype, "float32")
        self.assertEqual(complex_out.dtype, "complex64")
        self.assertEqual(train_calls, [False])
        self.assertIs(processor_target[0], fake_model)

    def test_move_inputs_to_model_casts_only_floating_tensors(self) -> None:
        class FakeTensor:
            def __init__(self, *, floating: bool) -> None:
                self.floating = floating
                self.calls: list[dict[str, object]] = []

            def is_floating_point(self) -> bool:
                return self.floating

            def to(self, **kwargs):
                self.calls.append(dict(kwargs))
                return ("moved", kwargs)

        class FakeModel:
            device = "cuda:0"
            dtype = "bfloat16"

        pixel_values = FakeTensor(floating=True)
        input_ids = FakeTensor(floating=False)

        moved = _move_inputs_to_model(
            {"pixel_values": pixel_values, "input_ids": input_ids, "meta": "keep"},
            FakeModel(),
        )

        self.assertEqual(pixel_values.calls, [{"device": "cuda:0", "dtype": "bfloat16"}])
        self.assertEqual(input_ids.calls, [{"device": "cuda:0"}])
        self.assertEqual(moved["meta"], "keep")


class TransformersProviderTests(unittest.TestCase):
    def test_build_provider_routes_detr(self) -> None:
        models = ModelConfig(provider="detr-resnet-50")
        provider = build_provider(models)
        self.assertIsInstance(provider, TransformersObjectDetectionProvider)
        self.assertEqual(provider.model_id, models.detr_model_id)

    def test_build_provider_routes_rt_detrv2(self) -> None:
        models = ModelConfig(provider="rt-detrv2")
        provider = build_provider(models)
        self.assertIsInstance(provider, TransformersObjectDetectionProvider)
        self.assertEqual(provider.model_id, models.rt_detr_model_id)

    def test_build_provider_rejects_unknown_provider(self) -> None:
        models = ModelConfig(provider="nope")
        with self.assertRaises(ValueError):
            build_provider(models)

    def test_matches_any_prompt_covers_coco_variants(self) -> None:
        self.assertTrue(_matches_any_prompt("potted plant", ["plant"]))
        self.assertTrue(_matches_any_prompt("dining table", ["table"]))
        self.assertTrue(_matches_any_prompt("person", ["person", "human"]))
        self.assertTrue(_matches_any_prompt("laptop", ["laptop"]))
        # normalize_label strips trailing plural "s"
        self.assertTrue(_matches_any_prompt("chairs", ["chair"]))
        # negative cases
        self.assertFalse(_matches_any_prompt("traffic light", ["chair"]))
        self.assertFalse(_matches_any_prompt("person", ["chair", "table"]))

    def test_transformers_provider_filters_by_prompt(self) -> None:
        provider = TransformersObjectDetectionProvider("test/model")

        scripted_result = {
            "scores": np.array([0.9, 0.8, 0.7, 0.6]),
            "labels": np.array([0, 1, 2, 3]),
            "boxes": np.array(
                [
                    [10.0, 20.0, 100.0, 200.0],
                    [150.0, 100.0, 300.0, 250.0],
                    [400.0, 300.0, 500.0, 400.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
        }

        class FakeProcessor:
            def __init__(self) -> None:
                self.post_process_calls: list[dict[str, object]] = []

            def __call__(self, images, return_tensors="pt"):
                return {"pixel_values": "sentinel"}

            def post_process_object_detection(
                self, outputs, threshold, target_sizes
            ):
                self.post_process_calls.append(
                    {
                        "threshold": threshold,
                        "target_sizes": target_sizes,
                    }
                )
                return [scripted_result]

        class FakeModel:
            def __init__(self) -> None:
                self.config = types.SimpleNamespace(
                    id2label={
                        0: "person",
                        1: "chair",
                        2: "potted plant",
                        3: "bicycle",
                    }
                )
                self.train_calls: list[bool] = []
                self.device = None
                self.call_log: list[dict[str, object]] = []

            def to(self, device):
                self.device = device
                return self

            def train(self, mode: bool) -> "FakeModel":
                self.train_calls.append(mode)
                return self

            def __call__(self, **kwargs):
                self.call_log.append(kwargs)
                return types.SimpleNamespace(outputs="sentinel")

        fake_processor = FakeProcessor()
        fake_model = FakeModel()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoImageProcessor = types.SimpleNamespace(
            from_pretrained=lambda model_id, trust_remote_code=False: fake_processor
        )
        fake_transformers.AutoModelForObjectDetection = types.SimpleNamespace(
            from_pretrained=lambda model_id, trust_remote_code=False: fake_model
        )

        class FakeInferenceMode:
            def __enter__(self) -> "FakeInferenceMode":
                return self

            def __exit__(self, *args) -> bool:
                return False

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

            class backends:
                class mps:
                    @staticmethod
                    def is_available() -> bool:
                        return False

            @staticmethod
            def device(name: str):
                return types.SimpleNamespace(type=name)

            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        with mock.patch.dict(
            sys.modules,
            {
                "transformers": fake_transformers,
                "torch": FakeTorch,
            },
        ):
            frame = VideoFrame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_index=0,
                source_id="test",
                fps=30.0,
            )
            detections = provider.detect_text_prompts(frame, ["chair", "plant"])

        self.assertEqual([d.label for d in detections], ["chair", "potted plant"])
        self.assertEqual([round(d.confidence, 2) for d in detections], [0.8, 0.7])
        self.assertTrue(all(d.mask is None for d in detections))
        self.assertEqual(fake_model.train_calls, [False])
        self.assertEqual(fake_processor.post_process_calls[0]["target_sizes"], [(480, 640)])

    def test_transformers_provider_reports_missing_dep(self) -> None:
        provider = TransformersObjectDetectionProvider("test/model")
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("No module named 'transformers'")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as exc:
                provider.warmup()

        message = str(exc.exception)
        self.assertIn("transformers", message)
        self.assertIn('.[desktop,detr]', message)

    def test_pyproject_defines_detr_extra(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        detr_extra = data["project"]["optional-dependencies"]["detr"]

        joined = "\n".join(detr_extra)
        self.assertIn("transformers", joined)
        self.assertIn("torch", joined)


if __name__ == "__main__":
    unittest.main()
