"""Select and explicitly enable a Cycles render device.

Run inside Blender. Arguments after ``--``:
  --device {auto,gpu,cpu} [--json PATH]
"""
import argparse
import json
import os
import sys
import traceback

import bpy


BACKENDS = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--json")
    return parser.parse_args(argv)


def _cycles_preferences():
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles add-on is unavailable")
    return addon.preferences


def enumerate_backends():
    prefs = _cycles_preferences()
    result = {}
    original = getattr(prefs, "compute_device_type", "NONE")
    for backend in BACKENDS:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            result[backend] = [
                {"name": d.name, "type": d.type, "id": d.id, "use": bool(d.use)}
                for d in prefs.devices
            ]
        except Exception as exc:  # backend may not be compiled in
            result[backend] = {"error": str(exc)}
    try:
        prefs.compute_device_type = original
    except Exception:
        pass
    return result


def configure(device_mode="auto"):
    prefs = _cycles_preferences()
    inventory = enumerate_backends()
    selected = "CPU"
    enabled = []

    if device_mode != "cpu":
        for backend in BACKENDS:
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
            except Exception:
                continue
            gpu_devices = [d for d in prefs.devices if d.type != "CPU"]
            if not gpu_devices:
                continue
            for d in prefs.devices:
                d.use = d in gpu_devices
            selected = backend
            enabled = [{"name": d.name, "type": d.type, "id": d.id} for d in gpu_devices]
            break

    if selected == "CPU":
        if device_mode == "gpu":
            raise RuntimeError("GPU rendering was explicitly requested, but Cycles exposed no supported GPU device")
        try:
            prefs.compute_device_type = "NONE"
        except Exception:
            pass
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "CPU"
        enabled = [{"name": d.name, "type": d.type, "id": d.id} for d in prefs.devices if d.type == "CPU"]

    for scene in bpy.data.scenes:
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "CYCLES" not in {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items} else "CYCLES"
        if scene.render.engine == "CYCLES":
            scene.cycles.device = "CPU" if selected == "CPU" else "GPU"

    info = {
        "requested": device_mode,
        "backend": selected,
        "enabled_devices": enabled,
        "available_backends": inventory,
    }
    print("CYCLES_DEVICE_JSON=" + json.dumps(info, sort_keys=True))
    return info


if __name__ == "__main__":
    try:
        arguments = _args()
        result = configure(arguments.device)
        if arguments.json:
            path = os.path.abspath(arguments.json)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(2)
