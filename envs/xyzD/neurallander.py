"""Neural-Lander: a quadrotor near the ground, where the aerodynamic force is a learned network.

State ``x = [p_x, p_y, p_z, v_x, v_y, v_z]``, control ``u`` the commanded acceleration.  The drift
term contains ``Fa(x)``, a pre-trained network (``model/Fa_net_12_3_full_Lip16.pth``) standing in
for the ground-effect force, which makes this the one benchmark whose true dynamics are not
analytic.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.env_base import BaseEnv

# --- physical parameters ---
drone_height, g, mass = 0.09, 9.81, 1.47

X_MIN = np.array([-20.0, -20.0, 0.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
X_MAX = np.array([20.0, 20.0, 5.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

XREF_INIT_MIN = np.array([-3.0, -3.0, 0.5, 1.0, 0.0, 0.0])
XREF_INIT_MAX = np.array([3.0, 3.0, 1.0, 1.0, 0.0, 0.0])
XE_INIT_MIN = np.array([-1.0, -1.0, -0.4, -1.0, -1.0, 0.0])
XE_INIT_MAX = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

lim = 1.0
XE_MIN = np.full((6, 1), -lim)
XE_MAX = np.full((6, 1), lim)

# The vertical channel has to be able to reach the hover command g - Fa_z / mass.
UREF_MIN = np.array([-1.0, -1.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 12.0]).reshape(-1, 1)

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
    "num_dim_control": 3,
    "pos_dimension": 3,
    "dt": 0.03,
    "time_bound": 6.0,
    "q": 1.0,
    "r": 0.0,
}

#: The Fa network predicts a normalised force; these restore physical units.
FA_SCALE = (30.0, 15.0, 10.0)
#: Rotor speed the pre-trained network was conditioned on, normalised by its maximum.
FA_RPM = 6508.0 / 8000.0


class FaNetwork(nn.Module):
    """The published ground-effect force network: 12 inputs -> 3 force components."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        """Predict the (normalised) aerodynamic force from the 12-dimensional feature vector."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


#: Pre-trained aerodynamic-force network, relative to the repository root.
FA_WEIGHTS = "model/Fa_net_12_3_full_Lip16.pth"


def load_fa_network(filename: str) -> FaNetwork:
    """Load the pre-trained weights, which were saved in double precision."""
    model = FaNetwork().double()
    model.load_state_dict(torch.load(filename, map_location="cpu"))
    return model.float().eval()


class NeuralLanderEnv(BaseEnv):
    """Quadrotor landing in ground effect, with the aerodynamic force given by a learned network.

    The only benchmark here whose true dynamics are not analytic, so it is the one that most
    directly tests the unknown-dynamics (``-approx``) setting.
    """

    task = "neurallander"

    REF_WEIGHT_SCALE = 0.5

    def __init__(
        self,
        sample_mode: str = "Uniform",
        reward_mode: str = "default",
        n_control_per_x: int = 1,
        ref_span: int | None = None,
        ref_stride: int = 1,
    ):
        # Absolute, resolved once here: the rollout workers reload the weights themselves (see
        # __setstate__) and must not depend on inheriting this process's working directory.
        self._fa_path = str((Path(__file__).resolve().parents[2] / FA_WEIGHTS))
        self.Fa_model = load_fa_network(self._fa_path)
        cfg = dict(
            env_config,
            sample_mode=sample_mode,
            reward_mode=reward_mode,
            n_control_per_x=n_control_per_x,
        )
        super().__init__(cfg, ref_span=ref_span, ref_stride=ref_stride)

    # ------------------------------------------------------------------ process boundary

    def __getstate__(self):
        """Drop the drag network before this environment is shipped to a rollout worker.

        The sampler sends the environment to its workers with the ``spawn`` start method, and
        multiprocessing pickles it with torch's ``ForkingPickler`` -- which reduces tensors by
        moving their storage into shared memory and passing the *file descriptor*.  Those
        descriptors are handed to ``fork_exec`` as ``fds_to_keep``, which rejects them with the
        entirely opaque ``ValueError: bad value(s) in fds_to_keep``.  This is why NeuralLander was
        the only benchmark that could not sample at all: it is the only environment that owns a
        network, so it is the only one with tensors to reduce.

        ``Fa`` is 1.4k frozen parameters on disk, so the child reloads it in :meth:`__setstate__`
        instead of sharing it.  Nothing measurable is lost, no descriptor crosses the boundary,
        and each worker gets a private read-only copy.
        """
        state = dict(self.__dict__)
        state.pop("Fa_model", None)
        return state

    def __setstate__(self, state):
        """Restore the environment in a worker process and reload its drag network."""
        self.__dict__.update(state)
        self.Fa_model = load_fa_network(self._fa_path)

    # ------------------------------------------------------------------ aerodynamics

    def Fa_func(self, x):
        """Aerodynamic force for a batch of states.

        Differentiable in ``x`` when given a torch tensor: the contraction conditions need
        ``df/dx``, and detaching this term would silently drop the only state-dependent part of the
        drift -- which for this system is the whole point of the benchmark.
        """
        if isinstance(x, np.ndarray):
            with torch.no_grad():
                return self._Fa(torch.as_tensor(x, dtype=torch.float32)).numpy()
        return self._Fa(x)

    def _Fa(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        z, v_x, v_y, v_z = x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        zeros, ones = torch.zeros_like(z), torch.ones_like(z)

        # Network input layout: [height, velocity (3), attitude quaternion (4), rotor speeds (4)].
        # Attitude is held level and the rotors at a fixed speed, as in the published model.
        features = torch.stack(
            [z + drone_height, v_x, v_y, v_z, zeros, zeros, zeros, ones]
            + [torch.full_like(z, FA_RPM)] * 4,
            dim=1,
        )

        self.Fa_model.to(device=x.device, dtype=x.dtype)
        scale = torch.tensor(FA_SCALE, dtype=x.dtype, device=x.device)
        return self.Fa_model(features) * scale

    # ------------------------------------------------------------------ dynamics

    def nominal_control(self, xref_0):
        """Hover command: cancel gravity and whatever aerodynamic force the reference sees."""
        Fa = self.Fa_func(np.asarray(xref_0, dtype=np.float32).reshape(1, -1)).flatten()
        return np.array([0.0, 0.0, g]) - Fa / mass

    def _f_logic(self, x, lib):
        Fa = self.Fa_func(x)
        v_x, v_y, v_z = x[:, 3], x[:, 4], x[:, 5]

        f = self._zeros(x, (x.shape[0], self.num_dim_x), lib)
        f[:, 0] = v_x
        f[:, 1] = v_y
        f[:, 2] = v_z
        f[:, 3] = Fa[:, 0] / mass
        f[:, 4] = Fa[:, 1] / mass
        f[:, 5] = Fa[:, 2] / mass - g
        return f

    def _B_logic(self, x, lib):
        B = self._zeros(x, (x.shape[0], self.num_dim_x, self.num_dim_control), lib)
        for i in range(self.num_dim_control):
            B[:, 3 + i, i] = 1.0
        return B
