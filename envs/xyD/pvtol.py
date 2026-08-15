"""Planar vertical-take-off-and-landing vehicle (PVTOL).

State ``x = [p_x, p_z, phi, v_x, v_z, phi_dot]`` in body-frame velocities, control ``u`` the two
rotor thrusts.  Trim (hover) thrust is ``m g / 2`` per rotor.
"""

import numpy as np

from envs.env_base import BaseEnv

# --- physical parameters ---
m, J, g, l = 0.486, 0.00383, 9.81, 0.25

p_lim = pd_lim = np.pi / 3
vx_lim, vz_lim = 2.0, 1.0

X_MIN = np.array([-15.0, 0.0, -p_lim, -vx_lim, -vz_lim, -pd_lim]).reshape(-1, 1)
X_MAX = np.array([15.0, 10.0, p_lim, vx_lim, vz_lim, pd_lim]).reshape(-1, 1)

XREF_INIT_MIN = np.array([-1.0, 5.0, -0.1, 0.5, 0.0, 0.0])
XREF_INIT_MAX = np.array([1.0, 6.0, 0.1, 1.0, 0.0, 0.0])
XE_INIT_MIN = np.full(6, -0.5)
XE_INIT_MAX = np.full(6, 0.5)

lim = 1.0
XE_MIN = np.full((6, 1), -lim)
XE_MAX = np.full((6, 1), lim)

UREF_MIN = np.full((2, 1), m * g / 2 - 1)
UREF_MAX = np.full((2, 1), m * g / 2 + 1)

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
    "num_dim_x": 6,
    "num_dim_control": 2,
    "pos_dimension": 2,
    "dt": 0.03,
    "time_bound": 6.0,
    "q": 1.0,
    "r": 0.0,
}


class PvtolEnv(BaseEnv):
    """PVTOL tracking a reference path while holding itself up on two rotors."""

    task = "pvtol"

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

    def nominal_control(self, xref_0):
        """Hover thrust: the two rotors together carry the weight."""
        return np.full(self.num_dim_control, 0.5 * m * g)

    def _f_logic(self, x, lib):
        _, _, phi, v_x, v_z, phi_dot = (x[:, i] for i in range(self.num_dim_x))

        f = self._zeros(x, (x.shape[0], self.num_dim_x), lib)
        f[:, 0] = v_x * lib.cos(phi) - v_z * lib.sin(phi)
        f[:, 1] = v_x * lib.sin(phi) + v_z * lib.cos(phi)
        f[:, 2] = phi_dot
        f[:, 3] = v_z * phi_dot - g * lib.sin(phi)
        f[:, 4] = -v_x * phi_dot - g * lib.cos(phi)
        return f

    def _B_logic(self, x, lib):
        B = self._zeros(x, (x.shape[0], self.num_dim_x, self.num_dim_control), lib)
        B[:, 4, 0] = 1 / m
        B[:, 4, 1] = 1 / m
        B[:, 5, 0] = l / J
        B[:, 5, 1] = -l / J
        return B
