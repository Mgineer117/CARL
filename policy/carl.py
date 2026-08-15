# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""CARL -- Contraction-Aware Reinforcement Learning.

For a control-affine system ``xdot = f(x) + B(x) u``, the shaped reward is

.. code-block:: text

    Phi(s) = 1 / (1 + q * e^T M(x) e)      in (0, 1]; equal to 1 exactly on the reference
    r_t    = gamma * Phi(s_{t+1}) - Phi(s_t)   potential-based shaping (Ng, Harada & Russell, 1999)

where ``M(x) = W(x)^-1`` is the learned metric and ``e = x - x*`` the tracking error.  Because
``Phi`` is bounded, the ``gamma * Phi(s') - Phi(s)`` form leaves the optimal policy unchanged for
any ``Phi``, which is what makes it safe to shape with a geometry that is itself still being
learned: a mis-specified metric can slow learning but cannot move the optimum.  CARL differs from
PPO in the geometry alone, which is what makes the comparison between them meaningful.  Only a
*learned* dynamics model is required, so the method applies when the true dynamics are unknown.

On the scope of that guarantee
------------------------------
Policy invariance concerns the exact discounted return, which here telescopes to
``gamma^T Phi(s_T) - Phi(s_0)``; at ``gamma^200 ~= 0.13`` it barely depends on the policy.  The
gradient in practice comes from GAE's finite effective horizon, over which the term acts as a dense
local-improvement signal.  The guarantee is best stated as the reason the shaping is *safe*, not as
the reason it works.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from policy.contraction import MetricLearner
from policy.ppo import PPO


class CARL(MetricLearner, PPO):
    """PPO whose reward is the tracking error measured in a *learned contraction metric*.

    The idea
    --------
    A control contraction metric ``M(x)`` certifies that any two trajectories of the closed loop
    approach each other exponentially when distance is measured as ``e^T M e``.  CARL learns that
    metric alongside the policy and uses it to score the policy through a *potential*

    .. code-block:: text

        Phi(s) = 1 / (1 + q * e^T M(x) e)          in (0, 1]; 1 exactly on the reference
        r_t    = gamma * Phi(s_{t+1}) - Phi(s_t)    potential-based shaping

    so the critic evaluates tracking error in the geometry under which the system actually
    contracts, rather than in the arbitrary Euclidean geometry of the raw state vector.  ``M`` is
    the *critic of the geometry*; the value function remains the critic of the return.

    The tracking term is **potential-based shaping** (Ng, Harada & Russell, 1999).  A reward of
    exactly the form ``gamma * Phi(s') - Phi(s)`` leaves the optimal policy unchanged for any
    bounded ``Phi``, which is what makes it safe to shape with a geometry that is still being
    learned: a mis-specified metric can slow learning but cannot move the optimum.  Scoring
    ``s_{t+1}`` rather than ``s_t`` credits the action for the state it actually produced.

    .. note::
       The guarantee is about the *exact* discounted return, and on these tasks that return is
       nearly vacuous: the sum telescopes to ``gamma^T Phi(s_T) - Phi(s_0)``, and at
       ``gamma^200 = 0.13`` it barely depends on the policy at all.  What actually supplies the
       gradient is GAE's finite effective horizon (~``1/(1-lambda)`` steps), over which the term
       is a dense local improvement signal.  So the invariance is worth stating as the reason the
       shaping is *safe*, not as the reason it *works*.

    Training alternates two objectives that share no parameters:

    * the **metric** minimises the contraction loss on offline states, with the current policy held
      fixed (``detach_policy=True``);
    * the **policy** runs ordinary PPO on the reward above, with the metric held fixed.

    The metric is updated once every ``metric_update_interval`` policy updates: it is the more expensive
    of the two and changes the reward under the critic's feet, so updating it less often keeps the
    value target comparatively stationary.

    Stochastic metric, deterministic reward
    ---------------------------------------
    The metric generator is stochastic (see :class:`~policy.layers.metrics.MetricNetwork`), and the
    two uses of it deliberately differ:

    * the **contraction loss samples** ``W``, so the conditions are enforced over a neighbourhood
      of metrics rather than at a single point estimate;
    * the **reward uses the mean**, so the quantity the critic must regress is not itself noisy.

    Args:
        x_dim: state dimension.
        ref_horizon: reference preview horizon carried in the observation.
        actor / critic: policy and value networks.
        metric_net: metric generator.
        get_f_and_B: dynamics oracle, analytic or learned.
        data: offline states for the contraction loss, from ``BaseEnv.sample_contraction_data``.
        metric_lr: metric learning rate.
        metric_batch_size: states per contraction update.
        metric_update_interval: policy updates between metric updates.
        warmup_epochs: metric-only pretraining updates on ``(C1)``, ``(C2)`` and the metric bounds.
        lbd / eps / w_lb / w_ub: contraction rate, constraint margin, and metric bounds.
        **ppo_kwargs: forwarded to :class:`~policy.ppo.PPO`, including the reward weights and the
            shaping form -- CARL contributes only the metric they are evaluated in.
    """

    name = "CARL"

    def __init__(
        self,
        x_dim: int,
        ref_horizon: int,
        actor: nn.Module,
        critic: nn.Module,
        metric_net: nn.Module,
        get_f_and_B: Callable,
        data: dict,
        metric_lr: float = 3e-4,
        metric_batch_size: int = 1024,
        metric_update_interval: int = 3,
        warmup_epochs: int = 1000,
        warmup_target: float = 0.01,
        lbd: float = 0.5,
        eps: float = 0.1,
        w_lb: float = 1e-1,
        w_ub: float = 10.0,
        num_probes: int = 1024,
        metric_entropy_scaler: float = 0.0,
        device: str = "cpu",
        **ppo_kwargs,
    ):
        # `tracking_scaler`, `control_scaler`, `reward_mode` and `reward_form` are PPO's: the whole
        # reward is defined there, and CARL changes only the metric it is measured in.
        super().__init__(
            x_dim=x_dim,
            ref_horizon=ref_horizon,
            actor=actor,
            critic=critic,
            device=device,
            **ppo_kwargs,
        )

        self.metric_update_interval = metric_update_interval

        self.setup_contraction(
            metric_net=metric_net,
            get_f_and_B=get_f_and_B,
            data=data,
            metric_lr=metric_lr,
            batch_size=metric_batch_size,
            lbd=lbd,
            eps=eps,
            w_lb=w_lb,
            w_ub=w_ub,
            num_probes=num_probes,
            metric_entropy_scaler=metric_entropy_scaler,
        )
        self.metric_lr_scheduler = LambdaLR(
            self.metric_optimizer, lr_lambda=self._metric_lr_multiplier
        )

        self.num_updates = 0
        self.to(self._dtype).to(self.device)
        self.warmup_metric(warmup_epochs, warmup_target)

    def _metric_lr_multiplier(self, _step):
        """Exponential decay to ~0.7% of the initial metric learning rate by the end of the run."""
        return float(np.exp(-5.0 * self.progress))

    # ------------------------------------------------------------------ reward

    def metric(self, x):
        """The learned contraction metric ``M(x) = W(x)^-1``.

        This is the *whole* of CARL's difference from PPO: the shaped reward, the potential, the
        terminal handling and the control term are all inherited unchanged, and PPO evaluates the
        same expression with ``M = I``.  The comparison between them is therefore an ablation of
        the geometry and of nothing else.
        """
        # The mean metric, not a sample: the reward is a regression target for the critic, and a
        # noisy target would be regressed as noise.
        _, M, _ = self.contraction.metric(x, deterministic=True)
        return M

    # ------------------------------------------------------------------ combined update

    def learn(self, batch: dict, progress: float):
        """One CARL update: optionally refresh the metric, then run a PPO update."""
        self.progress = progress
        loss_dict, supp_dict = {}, {}

        if self.num_updates % self.metric_update_interval == 0:
            loss, terms = self.metric_step(warmup=False)
            self.metric_lr_scheduler.step()
            loss_dict.update(
                self.metric_log(loss, terms, self.metric_lr_scheduler.get_last_lr()[0])
            )

            if self.num_updates % (100 * self.metric_update_interval) == 0:
                supp_dict.update(self.metric_plot())

        rl_losses, rl_supp, update_time = super().learn(batch, progress)
        loss_dict.update(rl_losses)
        supp_dict.update(rl_supp)

        self.num_updates += 1
        return loss_dict, supp_dict, update_time
