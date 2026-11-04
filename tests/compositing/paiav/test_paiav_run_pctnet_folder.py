from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


from reactbench.compositing.paiav import run_pctnet_folder as runner


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

    def test_completion_manifests_filter_partial_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            masks = root / "masks"
            completions = root / "completed_tokens"
            completions.mkdir(parents=True)
            for token in ("complete", "partial"):
                image_path = images / token / "camera" / "0.png"
                mask_path = masks / token / "camera" / "0.png"
                image_path.parent.mkdir(parents=True)
                mask_path.parent.mkdir(parents=True)
                cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))
                cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
            (completions / "complete.json").write_text(
                '{"status":"complete","token":"complete",'
                '"published_view_count":16,"expected_view_count":16}',
                encoding="utf-8",
            )

            completed_tokens = runner.load_completed_tokens(completions)
            items = runner.build_work_items(
                images,
                None,
                root / "final_images",
                completed_tokens,
            )

        self.assertEqual(completed_tokens, {"complete"})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].display_path.parts[0], "complete")

    def test_roi_harmonization_rejects_excessive_color_shift(self) -> None:
        image = np.full((100, 120, 3), 20, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[40:60, 50:60] = 255
        seen_shapes = []

        def model(crop: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            seen_shapes.append(crop.shape)
            return np.full_like(crop, 220)

        args = runner.parse_args(
            [
                "unused",
                "--roi-padding",
                "1.0",
                "--blend-strength",
                "0.5",
                "--max-mean-color-shift",
                "24",
            ]
        )
        result, quality = runner.harmonize_with_quality_gate(
            model, image, mask, args
        )

        self.assertFalse(quality["accepted"])
        self.assertIn(
            "excessive_mean_color_shift", quality["rejection_reasons"]
        )
        np.testing.assert_array_equal(result, image)
        self.assertLess(seen_shapes[0][0], image.shape[0])
        self.assertLess(seen_shapes[0][1], image.shape[1])

    def test_roi_harmonization_keeps_background_byte_identical(self) -> None:
        image = np.full((60, 80, 3), 40, dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[20:45, 30:50] = 255
        args = runner.parse_args(
            [
                "unused",
                "--roi-padding",
                "0.5",
                "--blend-strength",
                "0.5",
                "--max-mean-color-shift",
                "100",
            ]
        )

        result, quality = runner.harmonize_with_quality_gate(
            lambda crop, _mask: np.full_like(crop, 80),
            image,
            mask,
            args,
        )

        self.assertTrue(quality["accepted"])
        self.assertTrue(np.all(result[mask > 0] == 60))
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])


if __name__ == "__main__":
    unittest.main()
