"""Vectorized environments for parallel data collection and evaluation.

SyncVectorEnv: synchronous round-robin execution across sub-environments.
Each sub-env runs in its own process with an independent MuJoCo instance.
"""

import multiprocessing as mp
from typing import Any, Callable

import numpy as np


def _worker(env_fn: Callable[[], Any], seed: int, conn: mp.connection.Connection) -> None:
    """Worker process that owns one MuJoCo environment."""
    try:
        env = env_fn()
        env.reset(seed=seed)
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                action = data
                obs, reward, terminated, truncated, info = env.step(action)
                conn.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                seed_val = data
                obs, info = env.reset(seed=seed_val)
                conn.send((obs, info))
            elif cmd == "close":
                env.close()
                conn.send(None)
                break
            elif cmd == "get_obs":
                obs = env.get_observation()
                conn.send(obs)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


class SyncVectorEnv:
    """Synchronous vector environment running N independent MuJoCo instances.

    Each instance runs in its own process (multiprocessing) to avoid
    GL context / MuJoCo renderer conflicts.
    """

    def __init__(self, env_fns: list[Callable[[], Any]]):
        self._num_envs = len(env_fns)
        self._processes: list[mp.Process] = []
        self._parent_conns: list[mp.connection.Connection] = []
        self._child_conns: list[mp.connection.Connection] = []

        for i, env_fn in enumerate(env_fns):
            parent_conn, child_conn = mp.Pipe()
            self._parent_conns.append(parent_conn)
            self._child_conns.append(child_conn)
            p = mp.Process(target=_worker, args=(env_fn, i, child_conn), daemon=True)
            p.start()
            self._processes.append(p)

    def reset(self, seeds: list[int] | None = None) -> tuple[list[dict], list[dict]]:
        """Reset all environments. Returns (observations, infos)."""
        if seeds is None:
            seeds = list(range(self._num_envs))
        for i, conn in enumerate(self._parent_conns):
            conn.send(("reset", seeds[i]))
        results = [conn.recv() for conn in self._parent_conns]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions: list[np.ndarray]) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        """Step all environments with a list of actions."""
        for i, conn in enumerate(self._parent_conns):
            conn.send(("step", actions[i]))
        results = [conn.recv() for conn in self._parent_conns]

        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        terminated_list = [r[2] for r in results]
        truncated_list = [r[3] for r in results]
        info_list = [r[4] for r in results]
        return obs_list, reward_list, terminated_list, truncated_list, info_list

    def close(self) -> None:
        for conn in self._parent_conns:
            conn.send(("close", None))
        for conn in self._parent_conns:
            try:
                conn.recv()
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
