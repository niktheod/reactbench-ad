from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import run_pctnet_folder as runner  # noqa: E402


class PCTNetFolderRunnerTest(unittest.TestCase):
    @staticmethod
    def item(root: Path, name: str) -> runner.WorkItem:
        source = root / "source" / f"{name}.png"
        mask = root / "masks" / f"{name}.png"
        output = root / "output" / f"{name}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{name}".encode())
        mask.write_bytes(f"mask-{name}".encode())
        return runner.WorkItem(source, mask, output, Path(f"token/{name}.png"))

    def test_existing_output_is_skipped_before_decode_or_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            item = self.item(root, "cached")
            item.output_path.parent.mkdir(parents=True)
            item.output_path.write_bytes(b"existing")
            with patch.object(
                runner,
                "build_work_items",
                return_value=[item],
            ), patch.object(
                runner,
                "load_image_and_binary_mask",
            ) as decode, patch.object(runner, "load_pctnet") as load_model:
                result = runner.main(["unused", "--output-root", str(root / "output")])

        self.assertEqual(result, 0)
        decode.assert_not_called()
        load_model.assert_not_called()

    def test_pending_items_decode_once_and_model_loads_at_first_foreground(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cached = self.item(root, "cached")
            empty = self.item(root, "empty")
            foreground = self.item(root, "foreground")
            cached.output_path.parent.mkdir(parents=True)
            cached.output_path.write_bytes(b"existing")
            events: list[str] = []

            def decode(item: runner.WorkItem, _threshold: int):
                events.append(f"decode:{item.image_path.stem}")
                image = np.full((2, 3, 3), 20, dtype=np.uint8)
                mask = np.zeros((2, 3), dtype=np.uint8)
                foreground_pixels = 0
                if item is foreground:
                    mask[0, 0] = 255
                    foreground_pixels = 1
                return image, mask, runner.MaskStats(2, 3, 1, foreground_pixels)

            def load_model(_device):
                events.append("model:load")

                def model(image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
                    events.append("model:call")
                    return np.full_like(image, 200)

                return model

            with patch.object(
                runner,
                "build_work_items",
                return_value=[cached, empty, foreground],
            ), patch.object(
                runner,
                "load_image_and_binary_mask",
                side_effect=decode,
            ) as decode_mock, patch.object(
                runner,
                "load_pctnet",
                side_effect=load_model,
            ) as load_model_mock:
                result = runner.main(["unused", "--output-root", str(root / "output")])

            harmonized = cv2.imread(str(foreground.output_path), cv2.IMREAD_COLOR)
            copied_bytes = empty.output_path.read_bytes()

        self.assertEqual(result, 0)
        self.assertEqual(decode_mock.call_count, 2)
        load_model_mock.assert_called_once_with(0)
        self.assertEqual(
            events,
            [
                "decode:empty",
                "decode:foreground",
                "model:load",
                "model:call",
            ],
        )
        self.assertEqual(copied_bytes, b"source-empty")
        self.assertIsNotNone(harmonized)
        self.assertTrue(np.all(harmonized[0, 0] == 200))
        self.assertTrue(np.all(harmonized[0, 1] == 20))

    def test_dry_run_still_decodes_and_validates_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            items = [self.item(root, "one"), self.item(root, "two")]
            items[0].output_path.parent.mkdir(parents=True)
            items[0].output_path.write_bytes(b"existing")
            decoded = (
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
                runner.MaskStats(2, 2, 0, 0),
            )
            with patch.object(
                runner,
                "build_work_items",
                return_value=items,
            ), patch.object(
                runner,
                "load_image_and_binary_mask",
                return_value=decoded,
            ) as decode, patch.object(runner, "load_pctnet") as load_model:
                result = runner.main(
                    ["unused", "--output-root", str(root / "output"), "--dry-run"]
                )

        self.assertEqual(result, 0)
        self.assertEqual(decode.call_count, 2)
        load_model.assert_not_called()

    def test_reflink_or_copy_fallback_is_byte_identical_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            destination = root / "nested" / "destination.png"
            source.write_bytes(b"byte-identical-image-payload")

            method = runner.copy_image_atomic(source, destination)
            self.assertIn(method, {"reflink", "copy2"})
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            destination.write_bytes(b"changed-destination")
            self.assertEqual(source.read_bytes(), b"byte-identical-image-payload")

    def test_reflink_error_uses_copy2_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            destination = root / "destination.png"
            source.write_bytes(b"fallback-payload")

            with patch.object(
                runner.fcntl,
                "ioctl",
                side_effect=OSError("reflinks unsupported"),
            ):
                method = runner.copy_image_atomic(source, destination)

            self.assertEqual(method, "copy2")
            self.assertEqual(destination.read_bytes(), b"fallback-payload")


if __name__ == "__main__":
    unittest.main()
