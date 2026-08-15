# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Training loop for the online algorithms (CARL, PPO)."""

import os
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from trainer.evaluator import Evaluator


class OnlineTrainer(Evaluator):
    """Collect on-policy rollouts, update, evaluate periodically, checkpoint the best policy.

    Progress is measured in *environment steps*, so runs of different algorithms are compared at
    equal interaction budget.
    """

    def __init__(self, sampler, **evaluator_kwargs):
        super().__init__(**evaluator_kwargs)
        self.sampler = sampler

    def train(self):
        """Alternate rollout collection and policy updates until the step budget is spent."""
        start_time = time.time()
        total = self.epochs + self.init_epochs
        next_eval = 0
        # Advanced every collection so each batch draws *fresh* reference trajectories and initial
        # conditions.  Holding it fixed -- as this did -- makes the whole run train on the same
        # handful of episodes (14 workers x 3 = 42 for car) while evaluation uses disjoint seeds,
        # so the policy overfits those episodes and its measured tracking degrades even as its
        # training reward improves.  Still derived from `self.seed`, so runs stay reproducible.
        collect_seed = self.seed

        with tqdm(
            initial=self.init_epochs, total=total, desc=f"{self.policy.name} (env steps)"
        ) as pbar:
            while pbar.n < total:
                step = pbar.n + 1
                progress = step / total

                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=collect_seed
                )
                # Step past every episode seed this collection consumed, so no two collections can
                # overlap and the policy never trains on the same reference trajectory twice.
                collect_seed += self.sampler.num_envs
                loss_dict, supp_dict, update_time = self.policy.learn(batch, progress=progress)

                pbar.update(len(batch["rewards"]))

                elapsed = time.time() - start_time
                loss_dict.update(
                    {
                        # The step index is already the x-axis of every series, so it is not
                        # logged again; these are the two numbers you actually read mid-run.
                        f"{self.policy.name}/analytics/steps_per_sec": len(batch["rewards"])
                        / max(sample_time + update_time, 1e-9),
                        f"{self.policy.name}/analytics/hours_remaining": (
                            elapsed / progress - elapsed
                        )
                        / 3600,
                    }
                )
                self.write_log(loss_dict, step=step)
                self.write_image(supp_dict, step=step)

                if pbar.n >= next_eval or pbar.n >= total:
                    next_eval = pbar.n + self.eval_interval
                    eval_dict, eval_supp = self.evaluate()
                    is_best = self.record_best(eval_dict)
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(eval_supp, step=step)
                    self.save_model(step, eval_dict["eval/mauc"], is_best)

        self.logger.print(
            f"total {self.policy.name} training time: {(time.time() - start_time) / 3600:.2f} hours"
        )

    def save_model(self, step: int, mauc: float, is_best: bool = False):
        """Checkpoint the policy, and keep a copy of the best-scoring one so far.

        Everything needed to reproduce the run is saved -- for CARL that includes the learned
        metric ``metric_net``, which is the object the contraction claims are about and cannot be
        recovered from the actor alone.
        """
        state = {
            "step": step,
            "mauc": mauc,
            "actor": deepcopy(self.policy.actor).cpu().state_dict(),
        }
        for name in ("critic", "metric_net"):
            module = getattr(self.policy, name, None)
            if module is not None:
                state[name] = deepcopy(module).cpu().state_dict()

        torch.save(state, os.path.join(self.logger.checkpoint_dir, f"model_{step}.pth"))
        if is_best:
            torch.save(state, os.path.join(self.logger.log_dir, "best_model.pth"))
