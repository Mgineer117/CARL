"""Planar car (Dubins-like) with steerable heading and speed.

State ``x = [p_x, p_y, theta, v]``, control ``u = [theta_dot, v_dot]``.
"""

import numpy as np

from envs.env_base import BaseEnv

# --- speed limits ---
v_l, v_h = 1.0, 2.0

X_MIN = np.array([-15.0, -15.0, -np.pi, v_l]).reshape(-1, 1)
X_MAX = np.array([15.0, 15.0, np.pi, v_h]).reshape(-1, 1)

# Reference anchor and its perturbation into the initial state.
XREF_INIT_MIN = np.array([-2.0, -2.0, -1.0, 1.5])
XREF_INIT_MAX = np.array([2.0, 2.0, 1.0, 1.5])
XE_INIT_MIN = np.full(4, -1.0)
XE_INIT_MAX = np.full(4, 1.0)

# State perturbation range for the contraction dataset.
lim = 1.0
XE_MIN = np.full((4, 1), -lim)
XE_MAX = np.full((4, 1), lim)

UREF_MIN = np.array([-3.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([3.0, 3.0]).reshape(-1, 1)

env_config = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 4,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.03,
    "time_bound": 6.0,
    "q": 1.0,  # tracking cost weight
    "r": 0.0,  # control cost weight (0 = no control term)
}


class CarEnv(BaseEnv):
    """Planar car tracking a curved reference path at constant speed."""

    task = "car"

    #: The heading is the one circular state in this benchmark: its bounds are exactly [-pi, pi],
    #: so it must wrap.  Clipping it froze the heading for ~half of all training steps.
    ANGLE_DIMS = (2,)

    # Only the steering channel is excited, so the reference drives at a constant speed.
    REF_CONTROL_MASK = np.array([1.0, 0.0])

    def __init__(
        self,
        sample_mode: str = "Uniform",
        reward_mode: str = "default",
        n_control_per_x: int = 1,
        ref_span: int | None = None,
        ref_stride: int = 1,
    ):
        cfg = dict(
            env_config,
            sample_mode=sample_mode,
            reward_mode=reward_mode,
            n_control_per_x=n_control_per_x,
        )
        super().__init__(cfg, ref_span=ref_span, ref_stride=ref_stride)

    def _f_logic(self, x, lib):
        f = self._zeros(x, (x.shape[0], self.num_dim_x), lib)
        f[:, 0] = x[:, 3] * lib.cos(x[:, 2])
        f[:, 1] = x[:, 3] * lib.sin(x[:, 2])
        return f

    def _B_logic(self, x, lib):
        B = self._zeros(x, (x.shape[0], self.num_dim_x, self.num_dim_control), lib)
        B[:, 2, 0] = 1.0
        B[:, 3, 1] = 1.0
        return B
