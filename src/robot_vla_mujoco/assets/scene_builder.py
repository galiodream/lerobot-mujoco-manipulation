"""Scene builder: composes robot + arena + objects into a task XML.

For Milestone 1 this is a simple file-copy / path-resolution utility.
Full dynamic composition (MJCF include rewriting) will follow later.
"""

from pathlib import Path


def resolve_scene_path(scene_name: str, tasks_dir: str | Path = "assets/mujoco/tasks") -> Path:
    """Resolve a scene name to its XML file path."""
    tasks_dir = Path(tasks_dir)
    return tasks_dir / f"{scene_name}.xml"
