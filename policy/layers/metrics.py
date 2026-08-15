# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""The contraction metric generator (CMG): a state-dependent dual metric ``W(x)``."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from policy.layers.building_blocks import MLP


class MetricNetwork(nn.Module):
    r"""Learns the dual contraction metric ``W(x) = V(x)^T V(x)``, symmetric positive semi-definite.

    Two structural constraints, both taken from the reference implementation of C3M (Sun, Jha &
    Fan, CoRL 2020).  Neither is cosmetic: together they make ``(C2)`` hold *by construction*
    rather than by optimisation.

    Why the structure is load-bearing
    ---------------------------------
    For every benchmark here ``B`` drives only the trailing ``m`` states, so ``B = [0; I_m]`` and
    the annihilator is ``Bbot = [I_k; 0]`` with ``k = x_dim - action_dim``.  Left- and
    right-multiplying by ``Bbot`` therefore just *selects the top-left k x k block*, and the
    conditions read

    .. code-block:: text

        (C2)   d_{b_j} [W]_topleft  -  2 sym(dB_j/dx . W)_topleft   =  0      for each j
        (C1)   [ -d_f W + 2 sym(df/dx . W) + 2 lbd W ]_topleft      <  0

    ``W`` is built as ``V^T V`` from

    .. code-block:: text

        V = [ Wbot   V01 ]        Wbot: constant (its network is fed a vector of ones)
            [ 0      V11 ]        V01, V11: functions of the effective states only

    whose first ``k`` columns are ``[Wbot; 0]``, so ``[W]_topleft = Wbot^T Wbot`` is **constant in
    x** and ``d_{b_j} [W]_topleft = 0`` identically.  Where ``B`` is constant (the car, and every
    other system here), ``(C2)`` is then satisfied exactly, before a single gradient step.

    Without the structure, ``[W]_topleft`` is a generic function of the state and ``(C2)`` -- an
    *equality* -- becomes a squared-Frobenius penalty that gradient descent can shrink but never
    zero.  The residual then pulls against ``(Cu)``, ``(C1)`` and the overshoot term in a sum with
    nothing balancing it, so the minimiser of the loss is not the feasible point.  That is the
    failure mode this class previously had.

    ``W`` reads the whole state.  The reference implementation additionally restricts it to a
    per-system subset of coordinates (its ``effective_dim_start/end``), which for a
    translation-invariant system makes ``d_f W`` vanish identically as well; that is deliberately
    not done here, because it requires per-environment configuration.  The block structure above is
    system-agnostic -- it needs only ``x_dim`` and ``action_dim`` -- so ``(C2)`` still holds by
    construction, which is the part that matters.

    Args:
        x_dim: state dimension; ``W`` is ``(x_dim, x_dim)``.
        action_dim: control dimension, which fixes the ``k = x_dim - action_dim`` block split.
        hidden_dim: hidden widths of the trunk MLP.
        activation: trunk activation.  Must be twice differentiable -- the contraction loss takes
            second derivatives of ``W``, so a piecewise-linear activation such as ReLU would give a
            vanishing (and therefore meaningless) second-order term.
        stochastic: if ``True``, ``V`` is sampled from a learned diagonal Gaussian instead of being
            read off deterministically.  Note the resulting log-probability and entropy enter no
            loss, so the sampling acts purely as noise on the certificate -- and it makes the
            metric used to certify contraction differ from the (deterministic) metric used to shape
            the reward.  Prefer the default.
    """

    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.Tanh(),
        stochastic: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.action_dim = action_dim
        #: Size of the uncontrolled block that ``Bbot`` selects.
        self.k = x_dim - action_dim
        self.stochastic = stochastic

        self.trunk = MLP(x_dim, hidden_dim, activation=activation)
        self.mu = nn.Linear(hidden_dim[-1], x_dim * x_dim)
        self.logstd = nn.Linear(hidden_dim[-1], x_dim * x_dim) if stochastic else None

        # The constant block.  Its input is a vector of ones, so it is a learned constant rather
        # than a function of the state -- which is exactly what makes (C2) vanish.
        self.wbot = MLP(1, hidden_dim, self.k * self.k, activation=activation)

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        """Return ``(W, info)`` with ``W`` of shape ``(n, x_dim, x_dim)``, symmetric PSD."""
        n, d, k = x.shape[0], self.x_dim, self.k

        features = self.trunk(x)
        mu = self.mu(features)

        if self.stochastic and not deterministic:
            logstd = torch.clamp(self.logstd(features), min=-5.0, max=2.0)
            dist = Normal(loc=mu, scale=torch.exp(logstd))
            raw = dist.rsample()
            info = {
                "dist": dist,
                "logprobs": dist.log_prob(raw).sum(dim=-1, keepdim=True),
                "entropy": dist.entropy().sum(dim=-1, keepdim=True),
            }
        else:
            raw = mu
            zeros = torch.zeros_like(mu[:, :1])
            info = {"dist": None, "logprobs": zeros, "entropy": zeros}

        raw = raw.view(n, d, d)
        wbot = self.wbot(torch.ones(n, 1, dtype=x.dtype, device=x.device)).view(n, k, k)

        # Assembled by concatenation rather than in-place assignment into `raw`: writing into a
        # view of a tensor that carries gradient is exactly the pattern autograd refuses.  The
        # discarded blocks of `raw` are dead parameters, as they are in the reference code.
        top = torch.cat((wbot, raw[:, :k, k:]), dim=2)
        bottom = torch.cat((torch.zeros(n, d - k, k, dtype=x.dtype, device=x.device),
                            raw[:, k:, k:]), dim=2)
        V = torch.cat((top, bottom), dim=1)

        return V.transpose(1, 2) @ V, info
