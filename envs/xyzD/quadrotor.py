"""Quadrotor with thrust and attitude treated as states driven by their rates.

State ``x = [p_x, p_y, p_z, v_x, v_y, v_z, force, theta_x, theta_y, theta_z]``, control ``u`` the
rate of change of the last four states.
"""

import numpy as np

from envs.env_base import BaseEnv

g = 9.81

angle_lim = np.pi / 3
force_low, force_high = 0.5 * g, 2.0 * g
vel_lim = 1.5

X_MIN = np.array(
    [
        -30.0,
        -30.0,
        -30.0,
        -vel_lim,
        -vel_lim,
        -vel_lim,
        force_low,
        -angle_lim,
        -angle_lim,
        -angle_lim,
    ]
).reshape(-1, 1)
X_MAX = np.array(
    [30.0, 30.0, 30.0, vel_lim, vel_lim, vel_lim, force_high, angle_lim, angle_lim, angle_lim]
).reshape(-1, 1)

XREF_INIT_MIN = np.array([-5.0, -5.0, -5.0, -1.0, -1.0, -1.0, g, 0.0, 0.0, 0.0])
XREF_INIT_MAX = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0, g, 0.0, 0.0, 0.0])
XE_INIT_MIN = np.full(10, -0.5)
XE_INIT_MAX = np.full(10, 0.5)

lim = 1.0
XE_MIN = np.full((10, 1), -lim)
XE_MAX = np.full((10, 1), lim)

UREF_MIN = np.full((4, 1), -1.0)
UREF_MAX = np.full((4, 1), 1.0)

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
    "num_dim_x": 10,
    "num_dim_control": 4,
    "pos_dimension": 3,
    "dt": 0.03,
    "time_bound": 6.0,
    "q": 1.0,
    "r": 0.0,
}


class QuadRotorEnv(BaseEnv):
    """Quadrotor tracking a 3-D reference path; thrust and attitude are driven by their rates."""

    task = "quadrotor"

    REF_WEIGHT_SCALE = 0.1

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
        _, _, _, v_x, v_y, v_z, force, theta_x, theta_y, _ = (
            x[:, i] for i in range(self.num_dim_x)
        )

        f = self._zeros(x, (x.shape[0], self.num_dim_x), lib)
        f[:, 0] = v_x
        f[:, 1] = v_y
        f[:, 2] = v_z
        f[:, 3] = -force * lib.sin(theta_y)
        f[:, 4] = force * lib.cos(theta_y) * lib.sin(theta_x)
        f[:, 5] = g - force * lib.cos(theta_y) * lib.cos(theta_x)
        return f

    def _B_logic(self, x, lib):
        B = self._zeros(x, (x.shape[0], self.num_dim_x, self.num_dim_control), lib)
        for i in range(self.num_dim_control):
            B[:, 6 + i, i] = 1.0
        return B
