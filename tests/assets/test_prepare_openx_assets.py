from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARER_PATH = REPO_ROOT / "reactbench" / "assets" / "openx" / "prepare_openx_assets.py"
sys.path.insert(0, str(REPO_ROOT))
from reactbench.compositing.locations import BLENDER as BLENDER_PATH  # noqa: E402


def load_preparer_module():
    spec = importlib.util.spec_from_file_location("openx_asset_preparer", PREPARER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARER = load_preparer_module()


def valid_xoma() -> dict:
    return {
        "metadata": {
            "name": "Fixture Car",
            "description": "Synthetic preparation test vehicle.",
            "uuid": "6d3608c8-6777-54b7-a592-a3bc7130fb10",
            "authors": ["NAVSIM test fixture"],
            "license": "MPL-2.0 AND CC-BY-4.0",
            "copyrights": ["(c) NAVSIM test fixture"],
            "openMaterial3dVersion": "1.0.0",
            "assetVersion": "test",
            "assetType": "object",
            "objectClass": "vehicle",
            "boundingBox": {
                "x": [-2.0, 2.0],
                "y": [-1.0, 1.0],
                "z": [0.0, 1.5],
            },
            "vehicleClassData": {
                "vehicleCategory": "car",
                "axles": {
                    "frontAxle": {
                        "positionX": 1.2,
                        "positionZ": 0.3,
                        "trackWidth": 1.6,
                        "wheelDiameter": 0.6,
                    },
                    "rearAxle": {
                        "positionX": -1.2,
                        "positionZ": 0.3,
                        "trackWidth": 1.6,
                        "wheelDiameter": 0.6,
                    },
                },
            },
            "meshCount": 1,
            "triangleCount": 12,
            "animated": False,
        }
    }


def make_source_tree(tmp_path: Path, *, blend_bytes: bytes = b"BLENDER-v300"):
    root = tmp_path / "openx-assets"
    source = root / "src" / "vehicles" / "main" / "fixture_car"
    source.mkdir(parents=True)
    (root / "LICENSES").mkdir()
    (root / "LICENSES" / "MPL-2.0.txt").write_text("MPL fixture\n", encoding="utf-8")
    (root / "LICENSES" / "CC-BY-4.0.txt").write_text(
        "CC BY fixture\n", encoding="utf-8"
    )
    blend = source / "fixture_car.blend"
    xoma = source / "fixture_car.xoma"
    blend.write_bytes(blend_bytes)
    xoma.write_text(json.dumps(valid_xoma()), encoding="utf-8")
    return root, source.parent, blend, xoma


def test_metadata_preflight_accepts_complete_vehicle_pair(tmp_path):
    root, source_dir, blend, xoma = make_source_tree(tmp_path)
    assets = PREPARER.discover_assets(source_dir, root)

    assert [asset.stem for asset in assets] == ["fixture_car"]
    result = PREPARER.preflight_asset(assets[0], root)
    assert result["xoma"]["bounds_m"]["x"] == [-2.0, 2.0]
    assert result["xoma"]["axles"]["frontAxle"]["positionX"] == 1.2
    assert result["source_xoma_sha256"] == PREPARER.sha256_file(xoma)
    assert result["source_blend_sha256"] == PREPARER.sha256_file(blend)


def test_preflight_rejects_unresolved_git_lfs_pointer(tmp_path):
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:607a4ceedfe373bc1ec0dc5db4c9a4735afdc00b5a19c7e0cbaae06b524c4132\n"
        b"size 3235571\n"
    )
    root, source_dir, _, _ = make_source_tree(tmp_path, blend_bytes=pointer)
    asset = PREPARER.discover_assets(source_dir, root)[0]

    with pytest.raises(ValueError, match=r"Git LFS pointer.*3235571 bytes"):
        PREPARER.preflight_asset(asset, root)


def test_xoma_rejects_front_axle_behind_rear_axle(tmp_path):
    path = tmp_path / "bad.xoma"
    document = valid_xoma()
    document["metadata"]["vehicleClassData"]["axles"]["frontAxle"]["positionX"] = -1.3
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="frontAxle must be ahead of rearAxle"):
        PREPARER.load_xoma(path)


def test_xoma_allows_optional_unused_axle_position_z(tmp_path):
    path = tmp_path / "without_position_z.xoma"
    document = valid_xoma()
    for axle in document["metadata"]["vehicleClassData"]["axles"].values():
        axle.pop("positionZ")
    path.write_text(json.dumps(document), encoding="utf-8")

    parsed = PREPARER.load_xoma(path)
    assert parsed["axles"]["frontAxle"]["positionZ"] is None
    assert parsed["axles"]["rearAxle"]["positionZ"] is None


def test_discovery_rejects_orphan_xoma(tmp_path):
    root, source_dir, _, _ = make_source_tree(tmp_path)
    orphan = source_dir / "orphan"
    orphan.mkdir()
    (orphan / "orphan.xoma").write_text(json.dumps(valid_xoma()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing matching .blend"):
        PREPARER.discover_assets(source_dir, root)


def test_selection_supports_stem_globs_and_rejects_unmatched(tmp_path):
    root, source_dir, _, _ = make_source_tree(tmp_path)
    assets = PREPARER.discover_assets(source_dir, root)

    assert PREPARER.select_assets(assets, ["fixture_*"], None) == assets
    assert PREPARER.select_assets(
        assets, [], None, excluded_patterns=["fixture_*"]
    ) == []
    with pytest.raises(ValueError, match="No OpenX asset matched"):
        PREPARER.select_assets(assets, ["missing_*"], None)
    with pytest.raises(ValueError, match="matched exclusion"):
        PREPARER.select_assets(
            assets, [], None, excluded_patterns=["missing_*"]
        )


def test_known_nonconforming_assets_are_default_exclusions():
    def source_asset(stem: str):
        blend = Path("/openx/src/vehicles/main") / stem / f"{stem}.blend"
        return PREPARER.SourceAsset(
            blend=blend,
            xoma=blend.with_suffix(".xoma"),
            relative_blend=blend.relative_to("/openx"),
        )

    q7 = source_asset("m1_audi_q7_2015")
    mercedes = source_asset("m1_mercedes_sl65amg_2008")
    bmw = source_asset("m1_bmw_x1_2016")
    assets = [q7, mercedes, bmw]
    exclusions = PREPARER.default_exclusions_for_assets(assets)

    assert exclusions == ["m1_audi_q7_2015", "m1_mercedes_sl65amg_2008"]
    assert PREPARER.select_assets(
        assets, [], None, excluded_patterns=exclusions
    ) == [bmw]


def test_external_path_subtraction_preserves_non_weak_occurrences():
    weak = "/tmp/copybuffer.blend"
    texture = "/assets/paint.png"

    assert PREPARER.subtract_path_occurrences(
        [weak, weak, texture], [weak, weak]
    ) == [texture]
    assert PREPARER.subtract_path_occurrences([weak, weak], [weak]) == [weak]


def test_metadata_dry_run_works_without_blender_python(tmp_path):
    root, source_dir, _, _ = make_source_tree(tmp_path)
    output_dir = tmp_path / "output" / "assets"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARER_PATH),
            "--openx-root",
            str(root),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "OPENX_PREFLIGHT_COMPLETE selected=1" in completed.stdout
    assert not output_dir.exists()


@pytest.mark.skipif(not BLENDER_PATH.is_file(), reason="project Blender runtime is unavailable")
def test_blender_preparation_round_trip_is_packed_atomic_and_idempotent(tmp_path):
    root, source_dir, blend, xoma = make_source_tree(tmp_path)
    builder = tmp_path / "build_fixture.py"
    builder.write_text(
        """
import bpy
from pathlib import Path
import sys

output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
root = bpy.data.objects.new('Grp_Root', None)
scene.collection.objects.link(root)
bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.75))
body = bpy.context.object
body.name = 'VehicleBody'
body.scale = (2.0, 1.0, 0.75)
body.parent = root
material = bpy.data.materials.new('VehiclePaint')
body.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(BLENDER_PATH),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "2",
            "--python",
            str(builder),
            "--",
            str(blend),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_xoma_bytes = xoma.read_bytes()
    output_dir = tmp_path / "prepared" / "assets"
    command = [
        str(BLENDER_PATH),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "2",
        "--python",
        str(PREPARER_PATH),
        "--",
        "--openx-root",
        str(root),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)

    output_blend = output_dir / "fixture_car.blend"
    output_xoma = output_dir / "fixture_car.xoma"
    preparation = output_dir / "fixture_car.preparation.json"
    assert output_blend.read_bytes().startswith(PREPARER.BLENDER_COMPRESSED_HEADERS)
    assert output_xoma.read_bytes() == source_xoma_bytes
    metadata = json.loads(preparation.read_text(encoding="utf-8"))
    assert metadata["validation"]["passed"] is True
    assert metadata["validation"]["geometry_transformation_applied"] is False
    assert metadata["output"]["self_contained"] is True
    assert metadata["materials"]["ignored_library_weak_references"] == []
    assert "OPENX_PREPARATION_COMPLETE prepared=1 skipped=0 failed=0" in first.stdout
    manifest = json.loads(
        (output_dir.parent / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["included_asset_patterns"] == []
    assert manifest["excluded_asset_patterns"] == []
    assert manifest["default_exclusions_enabled"] is True
    attribution = (output_dir.parent / "OPENX_ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Fixture Car" in attribution
    assert "does not transform vehicle geometry" in attribution
    assert (output_dir.parent / "LICENSES" / "MPL-2.0.txt").is_file()
    assert (output_dir.parent / "LICENSES" / "CC-BY-4.0.txt").is_file()

    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "SKIP current fixture_car" in second.stdout
    assert "OPENX_PREPARATION_COMPLETE prepared=0 skipped=1 failed=0" in second.stdout
