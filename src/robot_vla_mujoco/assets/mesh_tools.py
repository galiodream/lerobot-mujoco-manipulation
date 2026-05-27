"""Mesh loading, inspection, and conversion utilities (stub for Milestone 1)."""

from pathlib import Path


def check_mesh_exists(mesh_path: str | Path) -> bool:
    return Path(mesh_path).exists()
