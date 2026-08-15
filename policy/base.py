# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Plumbing shared by every policy: dtype/device handling, observation parsing, log helpers."""

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.angles import wrap_angles


class BasePolicy(nn.Module, ABC):
    """Base class for all controllers in :mod:`policy`.

    Subclasses must set ``x_dim``, ``action_dim``, ``ref_horizon`` and ``device`` before calling
    :meth:`trim_state`, and must implement :meth:`learn`.
    """

    def __init__(self):
        super().__init__()
        self._dtype = torch.float32
        self.device = torch.device("cpu")
        self.mse_loss = F.mse_loss
        # Which states are circular.  Set by `register_angle_mask`; empty until then.
        self.register_buffer("angle_mask", torch.zeros(0, dtype=torch.bool), persistent=False)

    def register_angle_mask(self, angle_mask, x_dim: int):
        """Record which states live on a circle, so :meth:`trim_state` can wrap them."""
        mask = np.zeros(x_dim, dtype=bool) if angle_mask is None else np.asarray(angle_mask, bool)
        self.angle_mask = torch.as_tensor(mask, device=self.device)

    # ------------------------------------------------------------------ tensors

    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Move numpy data onto this policy's device in its working dtype."""
        return torch.as_tensor(data, dtype=self._dtype, device=self.device)

    def to_device(self, device):
        """Move the policy and remember where it lives, so :meth:`to_tensor` follows."""
        self.device = device
        self.to(device)

    # ------------------------------------------------------------------ pickling

    #: Attributes that exist only for training and are dropped when this policy is sent to a
    #: rollout worker.  Subclasses extend this; see :meth:`__getstate__`.
    TRAINING_ONLY_ATTRS: tuple[str, ...] = ()

    def __getstate__(self):
        """Strip training-only state before the policy is pickled to a rollout worker.

        A worker only ever calls :meth:`forward`.  Optimisers and learning-rate schedulers hold
        device-resident tensors that cannot cross a process boundary at all (sharing a non-CPU
        storage raises), and the offline datasets and dynamics oracle would be copied to every
        worker for nothing.  Dropping them keeps what a worker actually needs: the networks.
        """
        state = super().__getstate__()
        if not isinstance(state, dict):  # pragma: no cover - torch always returns a dict
            return state

        state = dict(state)
        for key, value in list(state.items()):
            training_only = key in self.TRAINING_ONLY_ATTRS or isinstance(
                value, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)
            )
            if training_only:
                state.pop(key)
        return state

    # ------------------------------------------------------------------ observation

    def trim_state(self, state: torch.Tensor):
        """Split a flat observation into its parts, with circular states wrapped.

        The layout is fixed by ``BaseEnv._observe``::

            [  x (x_dim)  |  xref[t:t+H] (H*x_dim)  |  uref[t] (action_dim)  ]

        where ``H = ref_horizon`` is the reference preview horizon.  ``H = 1`` reduces to a single
        reference point, which is what a non-preview controller uses.  There is no time
        index -- see :class:`~envs.env_base.BaseEnv`.

        Every state handed to an actor or a critic passes through here, so wrapping the angular
        components once at this boundary is what guarantees they never see an unwrapped angle,
        whatever produced the observation (the simulator, or the offline contraction dataset).

        Args:
            state: ``(n, state_dim)`` observations.

        Returns:
            ``(x, xref_seq, uref)`` with shapes ``(n, x_dim)``, ``(n, H, x_dim)`` and
            ``(n, action_dim)``.  The current reference point is ``xref_seq[:, 0]``; only the
            current reference *control* is carried, since nothing encodes a window of them.
        """
        n_x, n_u, H = self.x_dim, self.action_dim, self.ref_horizon

        x = state[:, :n_x]
        xref_seq = state[:, n_x : n_x + H * n_x].reshape(-1, H, n_x)
        uref = state[:, n_x + H * n_x : n_x + H * n_x + n_u]

        return wrap_angles(x, self.angle_mask), wrap_angles(xref_seq, self.angle_mask), uref

    # ------------------------------------------------------------------ logging

    def compute_gradient_norm(self, models, names, prefix: str = "", norm_type: float = 2):
        """L-``norm_type`` norm of the gradients of each model, keyed for the logger."""
        return {
            f"{prefix}/grad/{name}": _norm_of(
                (p.grad for p in model.parameters() if p.grad is not None), norm_type
            )
            for name, model in zip(names, models)
            if model is not None
        }

    @staticmethod
    def average_dict_values(dict_list: list[dict]) -> dict:
        """Mean of each key across a list of dicts (keys may be missing from some entries)."""
        if not dict_list:
            return {}
        keys = {k for d in dict_list for k in d}
        return {k: float(np.mean([d[k] for d in dict_list if k in d])) for k in keys}

    # ------------------------------------------------------------------ interface

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Run one training step and return ``(loss_dict, supp_dict, update_time)``."""


def _norm_of(tensors: Iterable[torch.Tensor], norm_type: float) -> float:
    tensors = [t for t in tensors]
    if not tensors:
        return 0.0
    per_tensor = torch.stack([torch.linalg.vector_norm(t.detach(), norm_type) for t in tensors])
    return torch.linalg.vector_norm(per_tensor, norm_type).item()
