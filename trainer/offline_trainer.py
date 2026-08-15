# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Training loop for the components fitted on offline data (currently the dynamics model)."""

import time
from copy import deepcopy

from tqdm import tqdm


class ComponentTrainer:
    """Fits an offline component -- the learned dynamics model -- by minibatch SGD.

    Args:
        env: environment supplying the dataset.
        module: component exposing ``learn(batch)`` and a ``name``.
        data: pre-sampled dataset exposing ``sample(batch_size)``.
        logger / writer: logging sinks.
        epochs: number of updates.
        batch_size: samples per update.
        init_epochs: offset so this component's curve continues the previous one's.
    """

    #: Updates between validation checks, and how many consecutive worsening checks are tolerated
    #: before stopping.  Patience above 1 because one check is noisy: the validation set is a
    #: subsample and the training batch is redrawn every step, so a single uptick is routine.
    VAL_INTERVAL = 250
    VAL_PATIENCE = 5

    def __init__(self, env, module, data, logger, writer, epochs, batch_size=1024, init_epochs=0):
        self.env = env
        self.module = module
        self.data = data
        self.logger = logger
        self.writer = writer
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_epochs = init_epochs

    def train(self):
        """Fit the component by minibatch SGD, stopping when held-out loss stops improving.

        ``epochs`` is a *budget*, not a target.  A component that has converged keeps training
        against a shrinking learning rate and eventually starts fitting the sample rather than the
        system, which for the dynamics model means handing the contraction conditions a certificate
        for dynamics that are subtly wrong.  Validation loss is what distinguishes the two, so the
        best-scoring weights are kept and restored at the end -- stopping early is not enough on its
        own, since the last iterate is not generally the best one.

        A component whose dataset exposes no held-out part simply runs its full budget.
        """
        start_time = time.time()
        can_validate = hasattr(self.data, "validation") and hasattr(self.module, "validation_loss")

        best_loss, best_state, best_step, stale = float("inf"), None, 0, 0
        stopped_early = False

        self.module.train()
        pbar = tqdm(range(1, self.epochs + 1), desc=f"{self.module.name} (epochs)")
        for step in pbar:
            loss_dict, update_time = self.module.learn(self.data.sample(self.batch_size))

            loss_dict[f"{self.module.name}/analytics/updates_per_sec"] = 1.0 / max(
                update_time, 1e-9
            )

            if can_validate and (step % self.VAL_INTERVAL == 0 or step == self.epochs):
                val = self.module.validation_loss(self.data.validation(8192))
                loss_dict[f"{self.module.name}/loss/val_mse"] = val
                pbar.set_postfix(val=f"{val:.3e}")

                if val < best_loss:
                    best_loss, best_step, stale = val, step, 0
                    best_state = deepcopy(self.module.state_dict())
                else:
                    stale += 1
                    if stale >= self.VAL_PATIENCE:
                        stopped_early = True

            global_step = self.init_epochs + step
            self.logger.store(**loss_dict)
            self.logger.write(global_step, eval_log=False, display=False)
            for key, value in loss_dict.items():
                self.writer.add_scalar(key, value, global_step)

            if stopped_early:
                pbar.close()
                break

        if best_state is not None:
            self.module.load_state_dict(best_state)
            reason = "validation loss stopped improving" if stopped_early else "budget spent"
            self.logger.print(
                f"[{self.module.name}] {reason}; restored step {best_step} "
                f"(val mse {best_loss:.4e}) after {step} of {self.epochs} updates."
            )

        self.module.eval()
        self.logger.print(
            f"total {self.module.name} training time: {(time.time() - start_time) / 3600:.2f} hours"
        )
