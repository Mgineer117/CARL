# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Proximal Policy Optimisation.

This is both the PPO baseline and the RL half of CARL, which subclasses it and overrides only the
reward (:mod:`policy.carl`).
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from policy.base import BasePolicy
from utils.functions import estimate_advantages


class PPO(BasePolicy):
    """Clipped-surrogate PPO with a Gaussian actor and a state-value critic.

    Conventional formulation: actor and critic share one Adam optimiser and one objective,

    .. code-block:: text

        L = L_clip  -  c1 * entropy  +  c2 * value_loss


    Args:
        x_dim: state dimension.
        ref_horizon: reference preview horizon carried in the observation.
        actor: policy network, ``(x, xref_seq, uref, deterministic) -> (a, info)``.
        critic: state-value network.
        rl_lr: Adam learning rate for the single optimiser over actor and critic.
        num_minibatch / minibatch_size: minibatches per epoch and their size.
        K: epochs over each collected batch.
        eps_clip: PPO ratio clipping range.
        entropy_scaler: weight of the entropy bonus (``c1``).
        value_scaler: weight of the value loss in the joint objective (``c2``).
        target_kl: stop the update once the sampled KL exceeds this.
        gamma / gae: discount and GAE trace decay.
        max_grad_norm: gradient clipping applied to the actor and critic.
        tracking_scaler / control_scaler: reward weights ``q`` and ``r``.
        reward_mode: how the control term is squashed, when there is one.
        reward_form: ``"potential"`` for ``gamma * Phi(s') - Phi(s)``, ``"next"`` for ``Phi(s')``.
        device: torch device.
    """

    name = "PPO"

    def __init__(
        self,
        x_dim: int,
        ref_horizon: int,
        actor: nn.Module,
        critic: nn.Module,
        rl_lr: float = 3e-4,
        num_minibatch: int = 4,
        minibatch_size: int = 2048,
        K: int = 10,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        value_scaler: float = 0.5,
        target_kl: float = 0.01,
        gamma: float = 0.99,
        gae: float = 0.95,
        max_grad_norm: float = 0.5,
        tracking_scaler: float = 1.0,
        control_scaler: float = 0.0,
        reward_mode: str = "inverse",
        reward_form: str = "potential",
        angle_mask=None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.ref_horizon = ref_horizon
        self.action_dim = actor.action_dim

        self.actor = actor
        self.critic = critic
        self.register_angle_mask(angle_mask, x_dim)

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.K = K
        self.eps_clip = eps_clip
        self.entropy_scaler = entropy_scaler
        self.value_scaler = value_scaler
        self.target_kl = target_kl
        self.gamma = gamma
        self.gae = gae
        self.max_grad_norm = max_grad_norm

        self.tracking_scaler = tracking_scaler
        self.control_scaler = control_scaler
        self.reward_mode = reward_mode
        if reward_form not in ("potential", "next"):
            raise ValueError(f"reward_form must be 'potential' or 'next', got {reward_form!r}")
        self.reward_form = reward_form

        # One optimiser over actor and critic, as in conventional PPO: a single loss
        # `actor - c1 * entropy + c2 * value` and a single step.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=rl_lr
        )

        self.progress = 0.0
        # Named method rather than a lambda: the policy is pickled to the rollout workers, and a
        # closure defined inside __init__ cannot be pickled.
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_multiplier)

        self.to(self._dtype).to(self.device)

    def _lr_multiplier(self, _step):
        """Linear decay to zero over the run, driven by environment steps rather than updates."""
        return max(0.0, 1.0 - self.progress)

    # ------------------------------------------------------------------ acting

    def forward(self, state: np.ndarray, deterministic: bool = False):
        """Act on a single observation (or a batch of them)."""
        state = self.to_tensor(np.asarray(state))
        if state.ndim == 1:
            state = state.unsqueeze(0)

        x, xref_seq, uref = self.trim_state(state)
        a, info = self.actor(x, xref_seq, uref, deterministic=deterministic)
        return a, {k: info[k] for k in ("probs", "logprobs", "entropy")}

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """Critic value of a batch of flat observations.

        The critic sees exactly the state and the reference path: not ``uref``, which is the
        actor's business, and not a time index.
        """
        x, xref_seq, _ = self.trim_state(states)
        return self.critic(x, xref_seq)

    # ------------------------------------------------------------------ training

    def learn(self, batch: dict, progress: float):
        """One PPO update over a collected batch."""
        self.train()
        t0 = time.time()
        self.progress = progress

        states = self.to_tensor(batch["states"])
        next_states = self.to_tensor(batch["next_states"])
        actions = self.to_tensor(batch["actions"])
        env_rewards = self.to_tensor(batch["rewards"])
        terminals = self.to_tensor(batch["terminals"])
        timeouts = self.to_tensor(batch["timeouts"])
        old_logprobs = self.to_tensor(batch["logprobs"])

        rewards, reward_info = self.shape_rewards(
            states, actions, env_rewards, next_states, terminals
        )

        with torch.no_grad():
            values = self.value(states)

            # `estimate_advantages` reads `next_values` only at rows where `timeouts > 0` -- one
            # row per episode, since these tasks always end by timeout.  Evaluating the critic on
            # the whole batch to use half a percent of it doubled the cost of the most expensive
            # forward pass in the update: the critic's GRU runs over the full H-step preview
            # window.  Only the rows that are read are computed; the rest stay zero and are never
            # looked at.
            next_values = torch.zeros_like(values)
            timeout_idx = torch.nonzero(timeouts.flatten() > 0, as_tuple=True)[0]
            if timeout_idx.numel():
                next_values[timeout_idx] = self.value(next_states[timeout_idx])

            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
                timeouts=timeouts,
                next_values=next_values,
            )

        stats, grad_dicts = _RunningStats(), []
        n = states.shape[0]
        epochs, kl_div = 0, torch.zeros(())

        # Conventional PPO: one joint objective, one optimiser, one step.  The KL guard ends the
        # whole update rather than the actor alone -- with shared parameters there is no longer a
        # meaningful way to keep training the critic while the actor sits out.
        for _ in range(self.K):
            stop = False
            for _ in range(self.num_minibatch):
                idx = torch.randperm(n, device=states.device)[: self.minibatch_size]

                actor_loss, entropy, clip_fraction, kl_div = self.actor_loss(
                    states[idx], actions[idx], old_logprobs[idx], advantages[idx]
                )
                if kl_div.item() > self.target_kl:
                    stop = True
                    break

                value_loss = self.critic_loss(states[idx], returns[idx])
                loss = actor_loss - self.entropy_scaler * entropy + self.value_scaler * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Actor and critic only: `self.parameters()` would also sweep in the metric and
                # the dynamics networks, whose gradients are not zeroed here and would silently
                # rescale this step.
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                grad_dicts.append(
                    self.compute_gradient_norm(
                        [self.actor, self.critic], ["actor", "critic"], prefix=self.name
                    )
                )
                stats.add(
                    total=loss.item(),
                    actor=actor_loss.item(),
                    critic=value_loss.item(),
                    entropy=entropy.item(),
                    clip_fraction=clip_fraction,
                    kl_divergence=kl_div.item(),
                )
            if stop:
                break
            epochs += 1

        self.lr_scheduler.step()

        # Losses under loss/, everything you would only *read* under analytics/.  Each quantity
        # appears once: the KL, the entropy and the clip fraction are diagnostics, not objectives,
        # even though they are computed inside the loss.
        means = stats.mean()
        loss_dict = {
            f"{self.name}/loss/total": means.get("total", 0.0),
            f"{self.name}/loss/actor": means.get("actor", 0.0),
            f"{self.name}/loss/critic": means.get("critic", 0.0),
            f"{self.name}/analytics/entropy": means.get("entropy", 0.0),
            f"{self.name}/analytics/clip_fraction": means.get("clip_fraction", 0.0),
            f"{self.name}/analytics/kl_divergence": means.get("kl_divergence", 0.0),
            f"{self.name}/analytics/epochs": epochs,
            f"{self.name}/analytics/reward": rewards.mean().item(),
            f"{self.name}/lr/rl": self.lr_scheduler.get_last_lr()[0],
        }
        loss_dict.update(reward_info)
        loss_dict.update(self.average_dict_values(grad_dicts))

        self.eval()
        return loss_dict, {}, time.time() - t0

    # ------------------------------------------------------------------ losses

    # ------------------------------------------------------------------ reward

    def metric(self, x: torch.Tensor) -> torch.Tensor:
        """The geometry in which tracking error is measured -- the identity for plain PPO.

        This is the *only* thing CARL overrides about the reward.  Everything below -- the bounded
        potential, the shaping form, the control term, the terminal handling -- is shared, so the
        two algorithms differ in the metric and in nothing else.  That is what makes the comparison
        between them an ablation rather than two separately-tuned reward functions: with ``M = I``
        the potential is the ordinary Euclidean ``1 / (1 + q * ||e||^2)``.
        """
        return torch.eye(self.x_dim, dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)

    def potential(self, states: torch.Tensor) -> torch.Tensor:
        """``Phi(s) = 1 / (1 + q * e^T M(x) e)`` -- how good it is to be at ``s``, in the metric.

        Bounded in ``(0, 1]``: 1 exactly on the reference, falling towards 0 as the error grows.
        The ``1 +`` is what bounds it -- ``1 / (e^T M e)`` alone diverges as the error goes to zero,
        which is precisely where a converging policy lives, and an unbounded potential makes the
        critic's regression target unbounded with it.
        """
        x, xref_seq, _ = self.trim_state(states)
        # The same wrapped error the environment and the actor use, so the shaped reward and the
        # reported tracking error cannot disagree about distance to the reference.
        error = self.actor.error(x, xref_seq[:, 0]).unsqueeze(-1)

        with torch.no_grad():
            energy = (error.transpose(1, 2) @ self.metric(x) @ error).squeeze(-1)
        return 1.0 / (1.0 + self.tracking_scaler * energy)

    def shape_rewards(self, states, actions, env_rewards, next_states, terminals):
        """The reward actually optimised, built from the potential above.

        Both forms score the state the transition *arrives in*, which is the state the action is
        actually responsible for:

        ``"next"``
            ``r_t = Phi(s_{t+1})``.  Direct, and the greedy thing to maximise: it rewards *being*
            close to the reference.

        ``"potential"``
            ``r_t = gamma * Phi(s_{t+1}) - Phi(s_t)``, potential-based shaping in the sense of Ng,
            Harada & Russell (1999).  A term of exactly this form leaves the optimal policy
            unchanged for *any* bounded ``Phi``, which is what makes it safe to shape with a
            potential that is still being learned.  The direct form has no such protection.

        The guarantee needs ``Phi = 0`` at true terminals, so the potential is zeroed there; that
        also means leaving the state bounds collects ``-Phi(s_t)``, forfeiting the accumulated
        position rather than merely stopping the reward stream.  A *timeout* is not a terminal --
        the episode continues, it is only the buffer that ends -- so those keep their potential,
        matching how ``estimate_advantages`` bootstraps them.
        """
        phi_next = self.potential(next_states) * (1.0 - terminals)
        if self.reward_form == "next":
            tracking = phi_next
        else:
            tracking = self.gamma * phi_next - self.potential(states)

        # Mirrors BaseEnv.get_rewards: a zero control scaler removes the term outright rather
        # than contributing a constant, which would only dilute the tracking signal.
        if self.control_scaler == 0:
            rewards = tracking
        else:
            control = -self.control_scaler * torch.linalg.norm(actions, dim=-1, keepdim=True)
            control = 1.0 / (1.0 + control.abs()) if self.reward_mode == "inverse" else control
            rewards = 0.5 * tracking + 0.5 * control

        # `analytics/reward` is the shaped reward being optimised; this is the environment's own
        # reward alongside it, so the two are comparable across algorithms.
        return rewards, {
            f"{self.name}/analytics/env_reward": env_rewards.mean().item(),
            f"{self.name}/analytics/potential": phi_next.mean().item(),
        }

    def actor_loss(self, states, actions, old_logprobs, advantages):
        """Clipped surrogate objective."""
        x, xref_seq, uref = self.trim_state(states)
        _, info = self.actor(x, xref_seq, uref)

        logprobs = self.actor.log_prob(info["dist"], actions)
        entropy = self.actor.entropy(info["dist"]).mean()
        ratios = torch.exp(logprobs - old_logprobs)

        surrogate = torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages,
        )

        with torch.no_grad():
            clip_fraction = ((ratios - 1).abs() > self.eps_clip).float().mean().item()
            kl_div = (old_logprobs - logprobs).mean()

        return -surrogate.mean(), entropy, clip_fraction, kl_div

    def critic_loss(self, states, returns):
        """Mean-squared value error against the GAE returns."""
        return self.mse_loss(self.value(states), returns)


class _RunningStats:
    """Accumulates scalar training statistics and averages them at the end of an update."""

    def __init__(self):
        self._values: dict[str, list[float]] = {}

    def add(self, **kwargs):
        """Record one minibatch's scalars."""
        for key, value in kwargs.items():
            self._values.setdefault(key, []).append(value)

    def mean(self) -> dict[str, float]:
        """Average each statistic over the minibatches recorded so far."""
        return {k: float(np.mean(v)) for k, v in self._values.items() if v}
