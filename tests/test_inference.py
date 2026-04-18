from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
import tomllib
import unittest
from unittest import mock

from meta_watcher.inference import CudaSam31Provider, _move_inputs_to_model


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


if __name__ == "__main__":
    unittest.main()
