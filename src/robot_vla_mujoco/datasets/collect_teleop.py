"""Keyboard teleop for data collection and debugging.

Usage (inside a rollout loop):
    from robot_vla_mujoco.datasets.collect_teleop import KeyboardTeleop
    teleop = KeyboardTeleop(env)
    action, done = teleop.get_action()
"""

import glfw
import numpy as np

from robot_vla_mujoco.mujoco_env.transforms import rpy2r, r2rpy
from robot_vla_mujoco.mujoco_env.utils import rotation_matrix


class KeyboardTeleop:
    """Keyboard-based teleoperation for the OMY arm.

    Keys:
        W/S       — move forward/backward (x)
        A/D       — move left/right (y)
        R/F       — move up/down (z)
        Q/E       — roll left/right
        UP/DOWN   — pitch
        LEFT/RIGHT — yaw
        SPACE     — toggle gripper
        Z         — reset episode
        ESC       — quit
    """

    def __init__(self, env):
        self._env = env  # MujocoManipulationEnv
        self._dpos_scale = 0.007
        self._drot_scale = 0.03
        self._gripper_state = True  # open

    def get_action(self) -> tuple[np.ndarray, bool]:
        """Returns (action_np, reset_episode_flag)."""
        env = self._env._env  # SimpleEnv2
        parser = env.env  # MuJoCoParserClass

        dpos = np.zeros(3)
        drot = np.eye(3)

        if parser.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-self._dpos_scale, 0.0, 0.0])
        if parser.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([self._dpos_scale, 0.0, 0.0])
        if parser.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0, -self._dpos_scale, 0.0])
        if parser.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0, self._dpos_scale, 0.0])
        if parser.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0, 0.0, self._dpos_scale])
        if parser.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0, 0.0, -self._dpos_scale])

        if parser.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=self._drot_scale, direction=[0.0, 1.0, 0.0])[:3, :3]
        if parser.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-self._drot_scale, direction=[0.0, 1.0, 0.0])[:3, :3]
        if parser.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=self._drot_scale, direction=[1.0, 0.0, 0.0])[:3, :3]
        if parser.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-self._drot_scale, direction=[1.0, 0.0, 0.0])[:3, :3]
        if parser.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=self._drot_scale, direction=[0.0, 0.0, 1.0])[:3, :3]
        if parser.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-self._drot_scale, direction=[0.0, 0.0, 1.0])[:3, :3]

        if parser.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True

        if parser.is_key_pressed_once(key=glfw.KEY_SPACE):
            self._gripper_state = not self._gripper_state

        drot_rpy = r2rpy(drot)
        action = np.concatenate(
            [dpos, drot_rpy, np.array([float(self._gripper_state)], dtype=np.float32)],
            dtype=np.float32,
        )
        return action, False
