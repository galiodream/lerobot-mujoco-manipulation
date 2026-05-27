"""pi0.5 policy adapter. Uses the same backend as pi0 (same model class)."""

from robot_vla_mujoco.policies.pi0_adapter import PI0Adapter


class PI05Adapter(PI0Adapter):
    """Adapter for pi0.5 VLA policy. Uses the same architecture as pi0."""

    pass
