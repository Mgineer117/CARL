# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Evaluation protocol and the metrics reported in the paper."""

import time
from abc import abstractmethod

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger

#: Colour-blind-safe qualitative palette, cycled over evaluation episodes.
COLORS = [
    "#4e79a7",
    "#f28e2c",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]


class Evaluator:
    """Runs the evaluation protocol and owns the logging shared by every trainer.

    **Protocol.**  ``eval_num`` reference trajectories are drawn; for each, ``eval_episodes``
    initial states are tried *against the same reference*, so the spread across episodes measures
    robustness to the initial tracking error rather than to the task.  Every evaluation uses the
    same fixed seed, so the same reference set is replayed at every logging step and across
    algorithms -- the training curve then reflects the policy improving, not the tasks changing.

    **Metrics.**  The headline number is ``mAUC``: the area under the *normalised* tracking-error
    curve ``||x(t) - xref(t)|| / ||x(0) - xref(0)||``.  It starts at 1 by construction, so it
    measures how fast the error is contracted, independent of how large the initial error happened
    to be, and it rewards fast transient decay rather than only the final error.  Episodes that end
    early are penalised by rescaling with the fraction of the horizon completed (``m2auc`` squares
    that factor, penalising early termination harder).
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        policy,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_epochs: int = 0,
        epochs: int = 10000,
        log_interval: int = 50,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
    ):
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.logger = logger
        self.writer = writer

        self.init_epochs = init_epochs
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = max(1, int(epochs / max(log_interval, 1)))

        self.eval_num = eval_num
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.best_mauc = float("inf")

    @abstractmethod
    def train(self):
        """Run the training loop."""

    @abstractmethod
    def save_model(self, step: int):
        """Checkpoint the policy."""

    # ------------------------------------------------------------------ evaluation

    def evaluate(self):
        """Run the full protocol and return ``(eval_dict, supp_dict)``."""
        self.policy.eval()
        per_reference, figure = [], None

        for i in tqdm(range(self.eval_num), desc="Evaluating", leave=False):
            episodes, tracks, errors = [], [], []

            for j in range(self.eval_episodes):
                # Seed only the FIRST episode of each reference.  That fixes the reference set
                # across calls, and the later episodes then draw successive initial conditions
                # from the advancing RNG.  Re-seeding on every `j` would rewind the generator to
                # the same state and redraw the *same* initial condition, silently collapsing
                # `--eval-episodes` into one episode repeated N times.
                options = {"replace_x_0": True} if j > 0 else None
                obs, _ = self.eval_env.reset(
                    seed=self.seed + i if j == 0 else None, options=options
                )
                episode, track, error = self._rollout(obs)
                episodes.append(episode)
                tracks.append(track)
                errors.append(error)

            if i == 0:
                figure = self.plot_trajectories(tracks, errors)

            summary = {k: float(np.mean([e[k] for e in episodes])) for k in episodes[0]}
            summary["overshoot"], summary["contraction_rate"] = self.contraction_rate(errors)
            per_reference.append(summary)

        # One value plus its confidence half-width per metric: `eval/<name>` is the number you
        # report, `eval/<name>_ci95` the interval around it.
        eval_dict = {}
        for key in per_reference[0]:
            mean, ci = self.mean_confidence_interval([r[key] for r in per_reference])
            eval_dict[f"eval/{key}"] = mean
            eval_dict[f"eval/{key}_ci95"] = ci

        return eval_dict, ({"eval/path_tracking_result": figure} if figure else {})

    def _rollout(self, obs):
        """One evaluation episode; returns ``(metrics, position_trace, normalised_error_trace)``."""
        env = self.eval_env
        horizon = env.episode_len

        ep_return, ctrl_effort, inf_time = 0.0, 0.0, 0.0
        # The error trace starts at t=0, where the normalised error is 1 by definition. Dropping
        # that point would shift every mAUC by ~dt and make this metric disagree with the one
        # reported by evaluate.py for the same rollout.
        track, error = [env.x_t[: env.pos_dimension].copy()], [1.0]

        for t in range(1, horizon + 1):
            with torch.no_grad():
                t0 = time.perf_counter()
                action, _ = self.policy(obs, deterministic=True)
                inf_time += time.perf_counter() - t0
            action = action.cpu().numpy().reshape(-1)

            obs, reward, terminated, truncated, infos = env.step(action)

            ep_return += self.policy.gamma**t * reward
            ctrl_effort += infos["control_effort"]
            track.append(infos["x"][: env.pos_dimension].copy())
            error.append(infos["relative_tracking_error"])

            if terminated or truncated:
                break

        auc = float(np.trapezoid(error, dx=env.dt))
        completion = horizon / t  # >= 1; larger when the episode ended early
        return (
            {
                "return": ep_return,
                "control_effort": ctrl_effort / t,
                "inference_time": inf_time / t,
                "mauc": auc * completion,
                "m2auc": auc * completion**2,
                "episode_len": float(t),
            },
            np.asarray(track),
            np.asarray(error),
        )

    # ------------------------------------------------------------------ metrics

    @staticmethod
    def contraction_rate(error_trajectories: list[np.ndarray]):
        """Fit ``||e(t)|| <= C exp(-lambda t) ||e(0)||`` to each episode; report ``(C, lambda)``.

        Each episode is fitted with its *own* overshoot ``C = max_t ||e(t)||/||e(0)|| >= 1``, so
        one bad episode cannot inflate the rate attributed to the others.  ``lambda`` is the
        largest rate consistent with every point of the episode, and the reported value is the
        median across episodes.
        """
        overshoots, rates = [], []

        for error in error_trajectories:
            error = np.maximum(np.asarray(error, dtype=float), 1e-12)
            C = max(float(error.max()), 1.0)
            overshoots.append(C)

            # lambda_t = ( log C - log e(t) ) / t, minimised over t > 0.
            t = np.arange(1, len(error)) * 1.0
            rates.append(max(0.0, float(np.min((np.log(C) - np.log(error[1:])) / t))))

        return float(np.mean(overshoots)), float(np.median(rates))

    @staticmethod
    def mean_confidence_interval(data):
        """Sample mean and half-width of its 95% CI (0 when there is a single sample).

        Fixed at 95%: the 1.96 is hard-coded, so a `confidence` argument would have been
        accepted and silently ignored.
        """
        data = np.asarray(data, dtype=float)
        mean = float(np.mean(data))
        if data.size < 2:
            return mean, 0.0
        return mean, float(1.96 * np.std(data, ddof=1) / np.sqrt(data.size))

    # ------------------------------------------------------------------ plotting

    def plot_trajectories(self, trajectories, error_trajectories):
        """Reference vs. tracked paths, alongside the normalised error curves."""
        dim = self.eval_env.pos_dimension
        xref = self.eval_env.xref

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d" if dim == 3 else None)
        ax2 = fig.add_subplot(1, 2, 2)

        def coords(path):
            """Split a path into per-axis coordinate arrays for plotting."""
            # A 1-D system has no path to draw, so plot its position against time instead.
            if dim == 1:
                return [np.arange(len(path)) * self.eval_env.dt, path[:, 0]]
            return [path[:, i] for i in range(dim)]

        ref = coords(xref)
        ax1.scatter(*[c[0] for c in ref], marker="*", c="black", s=80)
        ax1.plot(*ref, linewidth=2.0, linestyle="--", c="black", label="Reference")

        for i, path in enumerate(trajectories):
            c = COLORS[i % len(COLORS)]
            path = coords(np.asarray(path))
            ax1.scatter(*[p[0] for p in path], marker="*", alpha=0.9, c=c, s=80)
            ax1.plot(*path, alpha=0.9, c=c, label=f"episode {i}")
            ax2.plot(
                np.arange(len(error_trajectories[i])) * self.eval_env.dt, error_trajectories[i], c=c
            )

        if dim == 1:
            ax1.set_xlabel("time [s]", fontsize=14)
            ax1.set_ylabel("position", fontsize=14)
        else:
            ax1.set_xlabel("X", fontsize=14)
            ax1.set_ylabel("Y", fontsize=14)
            if dim == 3:
                ax1.set_zlabel("Z", fontsize=14)
                ax1.view_init(elev=25, azim=45)

        ax1.set_title("Path tracking", fontsize=16)
        ax2.set_xlabel("time [s]", fontsize=14)
        ax2.set_ylabel(r"$\|x(t)-x^*(t)\|_2\ /\ \|x(0)-x^*(0)\|_2$", fontsize=14)
        ax2.set_title("Normalised tracking error", fontsize=16)

        for ax in (ax1, ax2):
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------ logging

    def record_best(self, eval_dict: dict) -> bool:
        """Update the running best mAUC; return whether this evaluation improved on it.

        A single evaluation is noisy, so ``eval/mauc_best`` -- monotone by construction -- is the
        stable summary of a run and the right objective for a hyper-parameter search.  Optimising
        the last value instead lets the search chase evaluation noise.

        The returned flag is what decides the checkpoint, so the "is this the best so far?" test
        lives in exactly one place rather than being duplicated (and mis-ordered) per trainer.
        """
        improved = eval_dict["eval/mauc"] < self.best_mauc
        self.best_mauc = min(self.best_mauc, eval_dict["eval/mauc"])
        eval_dict["eval/mauc_best"] = self.best_mauc
        return improved

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        """Send scalars to WandB, TensorBoard and the CSV log."""
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, supp_dict: dict, step: int):
        """Send figures to WandB and release them, so long runs do not leak matplotlib state.

        Figure logging is best-effort.  These are diagnostics -- the contraction eigenvalue
        history, the trajectory plots -- and nothing downstream reads them, so a failure here must
        not take the run with it.  It did: wandb stages images through a temporary directory, and
        when that directory went missing under a sweep agent (several runs sharing ``/tmp`` on a
        compute node, one cleaning up while another was writing) the resulting ``FileNotFoundError``
        propagated out of ``learn`` and killed the trial.  Every sweep trial died this way before
        producing a single evaluation.

        The figure is closed either way, so the leak this method exists to prevent is unaffected.
        """
        for key, figure in supp_dict.items():
            try:
                self.logger.write_images(step=step, image=figure, logdir=key)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must never end a run
                print(f"[WARN] could not log figure {key!r} at step {step}: {exc}")
            finally:
                plt.close(figure)
