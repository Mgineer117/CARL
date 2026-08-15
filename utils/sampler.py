# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""On-policy rollout collection.

Environments are stepped in lockstep inside this process, so the policy is evaluated once per
timestep on the whole batch rather than once per environment.  That replaces ``N`` independent
batch-of-one forward passes -- each running a GRU over the full ``H``-step preview window, the
dominant cost of a step -- with a single batch-of-``N`` pass on the accelerator.  The environments
themselves stay in Python; their dynamics are a handful of numpy operations on a 4-to-10 element
state, which was never the bottleneck.
"""

import time
from copy import deepcopy
from math import ceil

import numpy as np
import torch

#: Keys every sampler returns, and the width of each row.
_FIELDS = ("states", "next_states", "actions", "rewards", "terminals", "timeouts", "logprobs")


class VectorSampler:
    """Steps a pool of in-process environments in lockstep, one batched policy call per timestep.

    Each environment rolls out exactly one episode per collection, so ``num_envs`` episodes come
    back per call and the transitions of one episode are contiguous -- which is what
    :func:`~utils.functions.estimate_advantages` needs, since it walks the buffer backwards and
    cuts traces at episode boundaries.

    Environments that end early (these tasks only terminate by leaving the state bounds) drop out
    of the batch, so the remaining ones do not pay for them.

    Args:
        state_dim: observation width.
        action_dim: control dimension.
        episode_len: steps per episode.
        batch_size: transitions requested per call; rounded up to whole episodes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        episode_len: int,
        batch_size: int,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.batch_size = batch_size

        #: One episode each, so this is both the environment count and the episodes per batch.
        self.num_envs = max(1, ceil(batch_size / episode_len))

        self._envs: list | None = None

        # The environments are numpy; letting torch grab every core for a batch-N forward on top of
        # that oversubscribes the node without helping.
        torch.set_num_threads(min(8, torch.get_num_threads()))

    # ------------------------------------------------------------------ pool

    def _pool(self, env):
        """Build the environment pool once and reuse it across collections."""
        if self._envs is None:
            # deepcopy rather than re-construction: an env may own loaded weights (NeuralLander's
            # drag network), and copying is both faster and guaranteed to match the original.
            self._envs = [deepcopy(env) for _ in range(self.num_envs)]
        return self._envs

    # ------------------------------------------------------------------ collection

    def collect_samples(self, env, policy, seed: int, deterministic: bool = False):
        """Roll out ``num_envs`` episodes and return ``(batch, elapsed_seconds)``."""
        t0 = time.time()
        envs = self._pool(env)
        was_training = policy.training
        policy.eval()

        # (env, step) buffers.  Episodes are written env-major so each one stays contiguous.
        buf = {
            "states": np.zeros((self.num_envs, self.episode_len, self.state_dim), np.float32),
            "next_states": np.zeros((self.num_envs, self.episode_len, self.state_dim), np.float32),
            "actions": np.zeros((self.num_envs, self.episode_len, self.action_dim), np.float32),
            "rewards": np.zeros((self.num_envs, self.episode_len, 1), np.float32),
            "terminals": np.zeros((self.num_envs, self.episode_len, 1), np.float32),
            "timeouts": np.zeros((self.num_envs, self.episode_len, 1), np.float32),
            "logprobs": np.zeros((self.num_envs, self.episode_len, 1), np.float32),
        }
        lengths = np.zeros(self.num_envs, dtype=int)

        obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
        live = np.arange(self.num_envs)  # indices of environments still running

        for t in range(self.episode_len):
            if live.size == 0:
                break

            # One forward pass for every environment still going, on whatever device the policy
            # already lives on -- no transfer to the CPU and back.
            with torch.no_grad():
                actions, info = policy(obs, deterministic=deterministic)
            actions = actions.cpu().numpy().reshape(len(live), self.action_dim)
            logprobs = info["logprobs"].cpu().numpy().reshape(len(live), 1)

            next_obs, still_live = [], []
            for row, env_idx in enumerate(live):
                nxt, reward, terminated, truncated, _ = envs[env_idx].step(actions[row])

                buf["states"][env_idx, t] = obs[row]
                buf["next_states"][env_idx, t] = nxt
                buf["actions"][env_idx, t] = actions[row]
                buf["rewards"][env_idx, t] = reward
                # `terminals` marks a true end of the return; a timeout is a *truncation* of an
                # otherwise-continuing episode and must not zero the bootstrap in GAE.
                buf["terminals"][env_idx, t] = float(terminated)
                buf["timeouts"][env_idx, t] = float(truncated)
                buf["logprobs"][env_idx, t] = logprobs[row]
                lengths[env_idx] = t + 1

                if not (terminated or truncated):
                    next_obs.append(nxt)
                    still_live.append(env_idx)

            obs = np.stack(next_obs) if next_obs else np.zeros((0, self.state_dim), np.float32)
            live = np.asarray(still_live, dtype=int)

        if was_training:
            policy.train()

        # Concatenate env-major and trim each episode to the length it actually ran, so no padding
        # reaches the buffer and every episode stays contiguous.
        batch = {
            key: np.concatenate([buf[key][i, : lengths[i]] for i in range(self.num_envs)])
            for key in _FIELDS
        }
        return batch, time.time() - t0


#: Backwards-compatible name.
OnlineSampler = VectorSampler
