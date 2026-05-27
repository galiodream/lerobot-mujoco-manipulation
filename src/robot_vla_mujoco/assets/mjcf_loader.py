"""MJCF loading and validation utilities."""

import os
import sys
from pathlib import Path

import mujoco


def load_model_from_xml(xml_path: str | Path) -> mujoco.MjModel:
    """Load a MuJoCo model from an XML path."""
    xml_path = str(xml_path)
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(xml_path)
    return mujoco.MjModel.from_xml_path(xml_path)


def validate_scene_xml(xml_path: str | Path) -> dict:
    """Validate a MuJoCo scene XML by loading it and returning diagnostic info.

    Returns a dict with keys: model_name, n_body, n_joint, n_geom, n_cam,
    n_ctrl, dt, integrator, body_names.
    """
    model = load_model_from_xml(xml_path)
    body_names = _get_body_names(model)
    return {
        "model_name": _model_name(model),
        "n_body": model.nbody,
        "n_joint": model.njnt,
        "n_geom": model.ngeom,
        "n_cam": model.ncam,
        "n_ctrl": model.nu,
        "dt": model.opt.timestep,
        "integrator": _integrator_name(model),
        "body_names": body_names,
    }


def _model_name(model: mujoco.MjModel) -> str:
    parsed = [s for s in model.names.split(b"\x00") if s]
    parsed = [s.decode("utf-8") for s in parsed]
    return parsed[0] if parsed else "unknown"


def _integrator_name(model: mujoco.MjModel) -> str:
    return {
        mujoco.mjtIntegrator.mjINT_EULER: "EULER",
        mujoco.mjtIntegrator.mjINT_RK4: "RK4",
        mujoco.mjtIntegrator.mjINT_IMPLICIT: "IMPLICIT",
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "IMPLICITFAST",
    }.get(model.opt.integrator, "UNKNOWN")


def _get_body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def validate_asset_root(asset_root: str | Path) -> list[dict]:
    """Scan an asset root directory for .xml files and validate each.

    Returns a list of validation result dicts.
    """
    results: list[dict] = []
    for root, _, files in os.walk(str(asset_root)):
        for fname in sorted(files):
            if fname.endswith(".xml"):
                full_path = os.path.join(root, fname)
                try:
                    info = validate_scene_xml(full_path)
                    info["path"] = os.path.relpath(full_path, asset_root)
                    info["status"] = "OK"
                except Exception as exc:
                    info = {
                        "path": os.path.relpath(full_path, asset_root),
                        "status": "FAIL",
                        "error": str(exc),
                    }
                results.append(info)
    return results


# CLI entry point for scripts/validate_assets.py
def main():
    asset_root = sys.argv[1] if len(sys.argv) > 1 else "assets/mujoco"
    print(f"Validating MuJoCo assets under: {asset_root}")
    results = validate_asset_root(asset_root)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    for r in results:
        if r["status"] == "OK":
            print(f'  OK  {r["path"]}  model={r["model_name"]} '
                  f'bodies={r["n_body"]} joints={r["n_joint"]} '
                  f'geoms={r["n_geom"]} cams={r["n_cam"]} ctrls={r["n_ctrl"]}')
        else:
            print(f'  FAIL {r["path"]}  error={r["error"]}')
    print(f"\n{n_ok} passed, {n_fail} failed out of {len(results)} XML files.")
    if n_fail:
        sys.exit(1)
