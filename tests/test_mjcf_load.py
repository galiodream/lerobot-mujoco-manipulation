"""Test that all MuJoCo XML assets can be loaded."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Discover all XML files under assets/mujoco, excluding include-only fragments
def _discover_xmls():
    asset_root = Path(__file__).resolve().parent.parent / "assets" / "mujoco"
    xmls = []
    for root, _, files in os.walk(asset_root):
        for f in sorted(files):
            if f.endswith(".xml"):
                full = os.path.join(root, f)
                # Skip include-only fragments (cannot be loaded standalone)
                if not _is_standalone_xml(full):
                    continue
                xmls.append(full)
    return xmls


def _is_standalone_xml(xml_path: str) -> bool:
    """Check if an XML file can be loaded as a standalone MuJoCo model."""
    with open(xml_path) as fh:
        head = fh.read(512)
    # Files with <mujocoinclude> are meant to be included, not loaded standalone
    if "<mujocoinclude" in head:
        return False
    # Include-only object fragments (reference parent-scene default classes)
    basename = os.path.basename(xml_path)
    if basename.startswith("obj_"):
        return False
    return True


XML_FILES = _discover_xmls()


@pytest.mark.parametrize("xml_path", XML_FILES)
def test_load_xml(xml_path):
    """Every .xml under assets/mujoco/ must be loadable by MuJoCo."""
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        pytest.fail(f"Failed to load {xml_path}: {e}")
    assert model is not None
    assert model.nbody > 0


def test_load_task_scene():
    """The main task scene must load correctly."""
    from robot_vla_mujoco.assets.mjcf_loader import validate_scene_xml

    task_xml = Path(__file__).resolve().parent.parent / "assets" / "mujoco" / "tasks" / "pick_place_mug.xml"
    info = validate_scene_xml(task_xml)
    assert info["n_body"] > 0
    assert info["n_joint"] > 0
    assert info["n_cam"] > 0
    assert info["n_ctrl"] > 0
