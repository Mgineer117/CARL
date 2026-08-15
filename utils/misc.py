# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Seeding, argument overriding, and logger setup."""

import json
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from log.wandb_logger import WandbLogger


def seed_all(seed: int = 0):
    """Seed every RNG the training loop draws from."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def setup_logger(args, unique_id: str, exp_time: str, seed: int, verbose: bool = True):
    """Create the WandB logger and the TensorBoard writer for one run."""
    if args.group is None:
        args.group = "-".join((exp_time, unique_id))
    if args.name is None:
        args.name = "-".join((args.algo_name, args.task, unique_id, f"seed:{seed}"))
    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)
    args.unique_id = unique_id

    config = vars(args)
    logger = WandbLogger(
        config=config,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
    )
    logger.save_config(config, verbose=verbose)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)
    return logger, SummaryWriter(log_dir=tensorboard_path)


def override_args(init_args):
    """Fill any argument left unset on the command line from the task's config file."""
    args = deepcopy(init_args)
    params = load_hyperparams(f"config/{args.task}.json")

    for key, value in params.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def load_hyperparams(file_path: str) -> dict:
    """Load a task config, coercing the step counts to ``int`` (JSON writes them as ``1e8``)."""
    try:
        with open(file_path) as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] No config at {file_path}; falling back to command-line defaults.")
        return {}

    for key in ("timesteps",):
        if key in params:
            params[key] = int(params[key])
    return params


def concat_csv_columnwise_and_delete(folder_path: str, output_file: str = "output.csv"):
    """Merge the per-seed CSV logs of one experiment group into a single file.

    Seeds of one group are normally submitted as *concurrent* jobs sharing a ``--group``, so this
    runs once per seed against a directory other seeds are still writing into.  Two rules keep
    that from destroying results:

    * ``output_file`` is never treated as an input.  It ends in ``.csv`` too, so a naive listing
      would merge the merged file into itself and then delete it -- leaving the group with no
      results at all once the second seed finished.
    * an existing merge is folded into the new one, so a seed finishing later extends the file
      rather than overwriting what earlier seeds already contributed.

    Only the per-seed files that were actually merged are removed; anything appearing mid-merge is
    left for the next call.
    """
    out_path = os.path.join(folder_path, output_file)
    inputs = sorted(f for f in os.listdir(folder_path) if f.endswith(".csv") and f != output_file)
    if not inputs:
        print("No CSV files found in the folder.")
        return

    frames = [pd.read_csv(out_path)] if os.path.exists(out_path) else []
    frames += [pd.read_csv(os.path.join(folder_path, f)) for f in inputs]
    # Column-wise: seeds may differ in length if one stopped early, and concat pads with NaN
    # rather than truncating every seed to the shortest.
    pd.concat(frames, axis=1).to_csv(out_path, index=False)

    for f in inputs:
        os.remove(os.path.join(folder_path, f))
    print(f"Combined CSV saved to {out_path}")
