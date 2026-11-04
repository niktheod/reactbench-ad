from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"
sys.path.insert(0, str(MODULE_ROOT))
import batch_composite_navsim_blender as compositor  # noqa: E402


class DepthPredictionCacheTest(unittest.TestCase):
    def test_legacy_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "workspace" / "dataset"
            source = (
                data_root
                / "sensor_blobs"
                / "test"
                / "log"
                / "CAM_F0"
                / "frame.jpg"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"unchanged source image bytes")
            source_sha256 = compositor.file_sha256(source)
            expected = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
            legacy = root / "run" / "source_relative_depth" / "frame.npz"
            legacy.parent.mkdir(parents=True)
            np.savez_compressed(
                legacy,
                relative_depth=expected,
                model_id=np.asarray(compositor.DEPTH_MODEL_ID),
                source_path=np.asarray(str(source.resolve())),
            )

            self.assertFalse(
                compositor.depth_prediction_matches(legacy, source_sha256)
            )

    def test_one_cached_source_materializes_all_token_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "workspace" / "dataset"
            source = (
                data_root
                / "sensor_blobs"
                / "test"
                / "log"
                / "CAM_F0"
                / "shared.jpg"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"shared source image bytes")
            source_sha256 = compositor.file_sha256(source)
            args = Namespace(
                depth_python=Path(sys.executable),
                data_root=data_root,
                depth_prediction_cache=root / "persistent_cache",
                depth_model_cache=root / "model_cache",
            )
            cache_path = compositor.depth_prediction_cache_path(
                args, source, source_sha256
            )
            expected = np.arange(20, dtype=np.float32).reshape(4, 5)
            cache_path.parent.mkdir(parents=True)
            np.savez_compressed(
                cache_path,
                relative_depth=expected,
                model_id=np.asarray(compositor.DEPTH_MODEL_ID),
                source_path=np.asarray(str(source.resolve())),
                source_sha256=np.asarray(source_sha256),
                cache_version=np.asarray(
                    compositor.DEPTH_PREDICTION_CACHE_VERSION
                ),
            )

            outputs = [
                root / "run" / "source_relative_depth" / token / "CAM_F0" / "0.npz"
                for token in ("token_a", "token_b")
            ]
            prepared = [
                {
                    "view_records": [
                        {
                            "projected_visible": True,
                            "source_path": str(source),
                            "relative_depth_path": str(output),
                        }
                    ]
                }
                for output in outputs
            ]
            run_dir = root / "run"
            run_dir.mkdir()
            compositor.run_depth_batch(args, run_dir, prepared)

            self.assertFalse((run_dir / "depth_jobs.jsonl").exists())
            for output in outputs:
                self.assertTrue(output.is_symlink())
                self.assertEqual(output.resolve(), cache_path.resolve())
                with np.load(output, allow_pickle=False) as cached:
                    np.testing.assert_array_equal(cached["relative_depth"], expected)


if __name__ == "__main__":
    unittest.main()
