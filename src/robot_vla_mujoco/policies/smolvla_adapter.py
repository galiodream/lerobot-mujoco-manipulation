"""SmolVLA policy adapter."""

from typing import Any

import numpy as np
import torch

from robot_vla_mujoco.policies.base import PolicyAdapter

try:
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    _HAS_SMOLVLA = True
except ImportError:
    _HAS_SMOLVLA = False


class SmolVLAAdapter(PolicyAdapter):
    """Adapter for SmolVLA (lightweight VLA model)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        action_dim: int = 7,
        image_size: int = 256,
        chunk_size: int = 10,
        n_action_steps: int = 5,
        dataset_stats: dict | None = None,
    ):
        if not _HAS_SMOLVLA:
            raise ImportError("lerobot SmolVLA not available. Install with: pip install lerobot")

        self._device = device
        self._action_dim = action_dim
        self._image_size = image_size
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps

        # Build a minimal config for inference
        cfg = SmolVLAConfig(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
        )

        self._policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path,
            config=cfg,
        )
        self._policy.to(device)
        self._policy.eval()

        # Image transform
        from torchvision import transforms
        self._img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

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

        front_img = observation.get("observation.images.front")
        wrist_img = observation.get("observation.images.wrist")
        task = observation.get("task", self._task)

        # Convert images to PIL if needed
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

        # select_action returns (batch, horizon, action_dim)
        if action.ndim == 3:
            action = action[0].cpu().numpy()
        else:
            action = action.cpu().numpy()

        return action

    def close(self) -> None:
        pass
