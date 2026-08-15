# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Learned control-affine dynamics, for the ``*-approx`` (unknown-dynamics) experiments."""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from policy.layers.building_blocks import MLP


class DynamicLearner(nn.Module):
    """Fits ``xdot = f(x) + B(x) u`` from ``(x, u, xdot)`` triples.

    The control-affine structure is imposed architecturally -- two heads producing ``f(x)`` and
    ``B(x)`` separately -- rather than regressing ``xdot`` from ``(x, u)`` jointly.  That is what
    lets the contraction conditions be written for the learned model at all: they need ``B`` and
    its annihilator as explicit objects, not just the composite prediction.

    Args:
        x_dim: state dimension.
        action_dim: control dimension.
        hidden_dim: hidden widths of both heads.
        nupdates: total updates, for the linear learning-rate decay.
        drop_out: dropout rate used during fitting.  Zero by default: the fit is now regularised
            by held-out validation and early stopping (see
            :class:`~trainer.offline_trainer.ComponentTrainer`), and dropout on a model whose
            *derivatives* are consumed by the contraction conditions is a poor trade -- it
            perturbs ``df/dx`` and ``dB/dx``, not just the prediction.
        activation: must be twice differentiable, since the contraction conditions differentiate
            ``f`` and ``B``.
    """

    name = "DynamicLearner"

    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        hidden_dim: list,
        nupdates: int,
        drop_out: float | None = None,
        activation: nn.Module = nn.Tanh(),
        dynamics_lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.action_dim = action_dim

        self.f = MLP(x_dim, hidden_dim, x_dim, activation=activation, dropout_rate=drop_out)
        self.B = MLP(
            x_dim, hidden_dim, x_dim * action_dim, activation=activation, dropout_rate=drop_out
        )

        self.nupdates = max(int(nupdates), 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=dynamics_lr)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_multiplier)

        self.device = device
        self.to(device)

    def _lr_multiplier(self, step):
        return max(0.0, 1.0 - step / self.nupdates)

    def forward(self, x):
        """``(f(x), B(x), Bbot(x))`` for a batch of states."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        f = self.f(x)
        B = self.B(x).view(-1, self.x_dim, self.action_dim)
        return f, B, B_annihilator(B)

    def learn(self, batch: dict):
        """One regression step on a batch of ``(x, u, xdot)``."""
        self.train()
        t0 = time.time()

        to_tensor = lambda k: torch.as_tensor(batch[k], dtype=torch.float32, device=self.device)
        x, u, x_dot = to_tensor("x"), to_tensor("u"), to_tensor("x_dot")

        f = self.f(x)
        B = self.B(x).view(-1, self.x_dim, self.action_dim)
        loss = F.mse_loss(f + (B @ u.unsqueeze(-1)).squeeze(-1), x_dot)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.eval()
        return {
            f"{self.name}/loss/mse": loss.item(),
            f"{self.name}/lr/dynamics": self.lr_scheduler.get_last_lr()[0],
        }, time.time() - t0

    @torch.no_grad()
    def validation_loss(self, batch: dict) -> float:
        """Mean-squared prediction error on held-out transitions.

        In ``eval()`` mode throughout, so the number is the model that will actually be handed to
        the contraction conditions -- not a dropout-perturbed one.
        """
        was_training = self.training
        self.eval()

        to_tensor = lambda k: torch.as_tensor(batch[k], dtype=torch.float32, device=self.device)
        x, u, x_dot = to_tensor("x"), to_tensor("u"), to_tensor("x_dot")
        B = self.B(x).view(-1, self.x_dim, self.action_dim)
        loss = F.mse_loss(self.f(x) + (B @ u.unsqueeze(-1)).squeeze(-1), x_dot)

        if was_training:
            self.train()
        return loss.item()


def B_annihilator(B: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the left null space of each ``B``, i.e. ``Bbot`` with ``B^T Bbot = 0``.

    The last ``x_dim - action_dim`` left-singular vectors of ``B`` span the orthogonal complement
    of its column space, which is what the ``C1``/``C2`` contraction conditions project onto.

    Args:
        B: ``(n, x_dim, action_dim)``.

    Returns:
        ``(n, x_dim, x_dim - action_dim)``.
    """
    U, _, _ = torch.linalg.svd(B)
    return U[:, :, B.shape[-1] :]
