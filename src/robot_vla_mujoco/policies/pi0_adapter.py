"""pi0 / pi0.5 policy adapter."""

from typing import Any

import numpy as np
import torch

from robot_vla_mujoco.policies.base import PolicyAdapter

try:
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    _HAS_PI0 = True
except ImportError:
    _HAS_PI0 = False


class PI0Adapter(PolicyAdapter):
    """Adapter for pi0 VLA policy."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        action_dim: int = 7,
        image_size: int = 256,
        chunk_size: int = 50,
        n_action_steps: int = 5,
        dataset_stats: dict | None = None,
    ):
        if not _HAS_PI0:
            raise ImportError("lerobot pi0 not available. Install with: pip install lerobot")

        self._device = device
        self._action_dim = action_dim
        self._image_size = image_size

        self._policy = PI0Policy.from_pretrained(
            checkpoint_path,
        )
        self._policy.config.chunk_size = chunk_size
        self._policy.config.n_action_steps = n_action_steps
        self._policy.to(device)
        self._policy.eval()

        from torchvision import transforms
        self._img_transform = transforms.Compose([transforms.ToTensor()])

        self._task = ""

    def reset(self, task: str) -> None:
        self._task = task
        self._policy.reset()

    def predict_action_trajectory(self, observation: dict[str, Any]) -> np.ndarray:
        import torch
        from PIL import Image

        state = observation.get("observation.state")
        if state is None:
            state = np.zeros(self._action_dim, dtype=np.float32)
        elif state.shape[-1] > self._policy.config.input_features["observation.state"].shape[0]:
            state = state[..., :self._policy.config.input_features["observation.state"].shape[0]]

        front_img = observation.get("observation.images.front")
        wrist_img = observation.get("observation.images.wrist")
        task = observation.get("task", self._task)

        if front_img is not None:
            if isinstance(front_img, np.ndarray):
                front_img = Image.fromarray(front_img)
            front_img = front_img.resize((self._image_size, self._image_size))
            front_tensor = self._img_transform(front_img)
        else:
            front_tensor = torch.zeros(3, self._image_size, self._image_size)

        if wrist_img is not None:
            if isinstance(wrist_img, np.ndarray):
                wrist_img = Image.fromarray(wrist_img)
            wrist_img = wrist_img.resize((self._image_size, self._image_size))
            wrist_tensor = self._img_transform(wrist_img)
        else:
            wrist_tensor = torch.zeros(3, self._image_size, self._image_size)

        batch = {
            "observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self._device),
            "observation.image": front_tensor.unsqueeze(0).to(self._device),
            "observation.wrist_image": wrist_tensor.unsqueeze(0).to(self._device),
            "task": [task],
        }

        with torch.inference_mode():
            action = self._policy.select_action(batch)

        if action.ndim == 3:
            action = action[0].cpu().numpy()
        else:
            action = action.cpu().numpy()

        return action

    def close(self) -> None:
        pass
