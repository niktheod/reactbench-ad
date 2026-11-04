from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import compact_run as compactor  # noqa: E402


class CompactRunTest(unittest.TestCase):
    token = "successful_token"
    failed_token = "failed_token"

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    @staticmethod
    def save_image(path: Path, array: np.ndarray, format_name: str = "PNG") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path, format=format_name)

    def build_run(self, root: Path) -> tuple[Path, dict[tuple[str, int], Path]]:
        run_dir = root / "run"
        source_root = root / "dataset"
        records = []
        sources: dict[tuple[str, int], Path] = {}
        for camera_index, camera in enumerate(compactor.CAMERAS):
            for frame_index in compactor.FRAME_INDICES:
                source_path = source_root / camera / f"{frame_index}.jpg"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_pixels = np.full(
                    (8, 10, 3),
                    35 + camera_index * 30 + frame_index,
                    dtype=np.uint8,
                )
                Image.fromarray(source_pixels).save(
                    source_path,
                    format="JPEG",
                    quality=92,
                    subsampling=0,
                )
                with Image.open(source_path) as source_image:
                    decoded_source = np.asarray(
                        source_image.convert("RGB"), dtype=np.uint8
                    ).copy()

                composite = decoded_source.copy()
                final = decoded_source.copy()
                mask = np.zeros(decoded_source.shape[:2], dtype=np.uint8)
                edited = camera == "CAM_F0" and frame_index == 3
                if edited:
                    mask[2, 4] = 255
                    composite[2, 4] = [120, 80, 40]
                    final[2, 4] = [125, 85, 45]

                relative = Path(self.token) / camera / f"{frame_index}.png"
                self.save_image(run_dir / "images" / relative, composite)
                self.save_image(run_dir / "masks" / relative, mask)
                self.save_image(run_dir / "actor_masks" / relative, mask)
                self.save_image(run_dir / "final_images" / relative, final)
                sources[(camera, frame_index)] = source_path
                records.append(
                    {
                        "camera": camera,
                        "frame_index": frame_index,
                        "source_path": str(source_path),
                        "mask_pixel_count": int(np.count_nonzero(mask)),
                        "changed_pixel_count": int(
                            np.count_nonzero(np.any(composite != decoded_source, axis=2))
                        ),
                    }
                )

        self.write_json(
            run_dir / "metadata" / f"{self.token}.json",
            {"status": "success", "token": self.token, "records": records},
        )
        self.write_json(
            run_dir / "failures" / f"{self.failed_token}.json",
            {"status": "failed", "token": self.failed_token},
        )
        self.write_json(
            run_dir / "summary.json",
            {
                "selected_tokens": 2,
                "successful_tokens": 1,
                "failed_tokens": 1,
                "potential_failure_tokens": 0,
            },
        )
        self.write_json(
            run_dir / "run_config.json",
            {"scene_filter": "/filter.yaml", "data_root": "/dataset"},
        )
        self.write_json(
            run_dir / "shadow_labels.json",
            {"document_role": "direct_shadow_labels", "labels": {}},
        )
        self.write_json(run_dir / "potential_failures.json", [])
        self.write_json(run_dir / "annotations.json", [])
        (run_dir / "depth_inference.log").write_text("complete\n")

        partial = np.zeros((8, 10, 3), dtype=np.uint8)
        self.save_image(
            run_dir
            / "final_images"
            / self.failed_token
            / "CAM_F0"
            / "0.png",
            partial,
        )
        for name in compactor.CLEANUP_DIRECTORIES:
            directory = run_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            if name not in {"images", "masks", "actor_masks"}:
                (directory / "artifact.bin").write_bytes(b"temporary")
        return run_dir, sources

    def test_compaction_links_unchanged_frames_and_removes_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, sources = self.build_run(Path(temporary_directory))

            report = compactor.compact_run(run_dir, workers=2)

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["linked_original_jpeg_frames"], 11)
            self.assertEqual(report["retained_edited_png_frames"], 1)
            for camera in compactor.CAMERAS:
                for frame_index in compactor.FRAME_INDICES:
                    stem = run_dir / "final_images" / self.token / camera / str(frame_index)
                    if camera == "CAM_F0" and frame_index == 3:
                        self.assertTrue(stem.with_suffix(".png").is_file())
                        self.assertFalse(stem.with_suffix(".jpg").exists())
                    else:
                        linked = stem.with_suffix(".jpg")
                        self.assertTrue(linked.is_symlink())
                        self.assertEqual(
                            linked.resolve(), sources[(camera, frame_index)].resolve()
                        )
                        self.assertFalse(stem.with_suffix(".png").exists())

            self.assertFalse((run_dir / "final_images" / self.failed_token).exists())
            for name in (*compactor.CLEANUP_DIRECTORIES, *compactor.CLEANUP_FILES):
                self.assertFalse((run_dir / name).exists(), name)
            for retained in (
                "metadata",
                "run_config.json",
                "summary.json",
                "failures",
                "potential_failures.json",
                "shadow_labels.json",
                compactor.REPORT_FILENAME,
            ):
                self.assertTrue((run_dir / retained).exists(), retained)

            second_report = compactor.compact_run(run_dir, workers=2)
            self.assertEqual(second_report["status"], "complete")

    def test_validation_failure_deletes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, _ = self.build_run(Path(temporary_directory))
            bad_final = (
                run_dir
                / "final_images"
                / self.token
                / "CAM_L0"
                / "0.png"
            )
            with Image.open(bad_final) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            pixels[0, 0] = [255, 0, 0]
            self.save_image(bad_final, pixels)

            with self.assertRaisesRegex(
                compactor.CompactionError,
                "no files were deleted",
            ):
                compactor.compact_run(run_dir, workers=2)

            self.assertTrue((run_dir / "images").is_dir())
            self.assertTrue((run_dir / "masks").is_dir())
            self.assertTrue((run_dir / "raw_ideal").is_dir())
            self.assertTrue(bad_final.is_file())
            self.assertFalse(
                bad_final.with_suffix(".jpg").exists()
                or bad_final.with_suffix(".jpg").is_symlink()
            )
            self.assertFalse((run_dir / compactor.REPORT_FILENAME).exists())

    def test_validate_only_performs_no_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, _ = self.build_run(Path(temporary_directory))

            report = compactor.compact_run(
                run_dir,
                workers=2,
                validate_only=True,
            )

            self.assertEqual(report["status"], "validated_only")
            self.assertTrue((run_dir / "images").is_dir())
            self.assertTrue((run_dir / "masks").is_dir())
            self.assertEqual(
                len(list((run_dir / "final_images" / self.token).rglob("*.png"))),
                12,
            )
            self.assertFalse((run_dir / compactor.REPORT_FILENAME).exists())

    def test_interrupted_publication_is_marked_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, _ = self.build_run(Path(temporary_directory))
            with patch.object(
                compactor,
                "publish_compact_final_images",
                side_effect=OSError("simulated publication interruption"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    compactor.compact_run(run_dir, workers=2)

            report = json.loads(
                (run_dir / compactor.REPORT_FILENAME).read_text()
            )
            self.assertEqual(report["status"], "publication_pending")
            self.assertTrue((run_dir / "images").is_dir())
            self.assertTrue((run_dir / "masks").is_dir())

            completed = compactor.compact_run(run_dir, workers=2)
            self.assertEqual(completed["status"], "complete")


if __name__ == "__main__":
    unittest.main()
