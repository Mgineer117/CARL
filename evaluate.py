# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Evaluate a trained checkpoint and report the tracking metrics.

    python evaluate.py --task car --algo-name carl --checkpoint log/train_log/.../best_model.pth

Checkpoints are the dicts written by the trainers (``actor`` / ``critic`` / ``metric_net`` state dicts
plus the step and score they were saved at), so a run can be reproduced from its own log directory
without moving files around.
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch

from utils.get_args import int_list, select_device, training_defaults
from utils.misc import load_hyperparams, seed_all
from utils.utils import call_env, get_policy, parse_algo_name

#: Training knobs that must differ when we are only evaluating.  The metric and the controller come
#: from the checkpoint, so there is nothing to warm up and no offline dataset worth building.
EVALUATION_OVERRIDES = dict(
    warmup_epochs=0,
    contraction_buffer_size=2,
    dynamics_buffer_size=2,
    dynamics_epochs=1,
    timesteps=1,
)


def parse_args():
    """Parse the command line, filling architecture defaults so they match the checkpoint."""
    p = argparse.ArgumentParser(description="Evaluate a trained CARL checkpoint.")
    p.add_argument("--task", type=str, default="car")
    p.add_argument("--algo-name", type=str, default="carl")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a saved *.pth file.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-num", type=int, default=10, help="Reference trajectories.")
    p.add_argument("--eval-episodes", type=int, default=10, help="Initial states per reference.")
    p.add_argument("--gpu-idx", type=int, default=0)
    p.add_argument(
        "--device", type=str, default=None, help="Force a device: cuda | cuda:N | mps | cpu."
    )
    p.add_argument("--output", type=str, default=None, help="Optional path for a JSON summary.")

    # Architecture: must match the checkpoint.  These default to None so the *training* defaults
    # apply unless you actually pass one -- restating the values here is what lets the two entry
    # points drift apart.
    p.add_argument(
        "--ref-span",
        type=int,
        default=None,
        help="Must match the checkpoint; defaults to the full episode length.",
    )
    p.add_argument("--ref-stride", type=int, default=None, help="Must match the checkpoint.")
    p.add_argument("--ref-dim", type=int, default=None, help="Must match the checkpoint.")
    p.add_argument("--actor-dim", type=int_list, default=None, help="Must match the checkpoint.")
    p.add_argument("--critic-dim", type=int_list, default=None, help="Must match the checkpoint.")
    p.add_argument("--metric-dim", type=int_list, default=None, help="Must match the checkpoint.")

    args = p.parse_args()

    # Start from the *training* defaults rather than restating them, so renaming or adding a
    # training flag can never leave this entry point silently out of date.  Precedence, lowest
    # first: training defaults -> task config -> evaluation overrides -> this command line.
    merged = training_defaults()
    merged.update(load_hyperparams(f"config/{args.task}.json"))
    merged.update(EVALUATION_OVERRIDES)
    merged.update({key: value for key, value in vars(args).items() if value is not None})
    merged.setdefault("output", None)

    resolved = SimpleNamespace(**merged)
    resolved.device = select_device(args.gpu_idx, requested=args.device, verbose=False)
    return resolved


def load_policy(args, env):
    """Rebuild the policy and restore its weights from the checkpoint."""
    policy = get_policy(env, args, env.get_f_and_B)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Accept both the structured checkpoints and a bare actor state dict.
    if not isinstance(state, dict) or "actor" not in state:
        state = {"actor": state}

    for name in ("actor", "critic", "metric_net"):
        module = getattr(policy, name, None)
        if module is not None and name in state:
            module.load_state_dict(state[name])

    if "step" in state:
        print(
            f"[INFO] Loaded {args.checkpoint} (step {state['step']}, mAUC {state.get('mauc'):.4f})"
        )
    policy.to_device(args.device)
    policy.eval()
    return policy


def main():
    """Load a checkpoint, run the evaluation protocol, and print (optionally save) the metrics."""
    args = parse_args()
    seed_all(args.seed)
    parse_algo_name(args.algo_name)  # fail fast on a typo

    env = call_env(args)
    policy = load_policy(args, env)

    from trainer.evaluator import Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.eval_env, evaluator.policy = env, policy
    evaluator.eval_num, evaluator.eval_episodes, evaluator.seed = (
        args.eval_num,
        args.eval_episodes,
        args.seed,
    )
    eval_dict, _ = evaluator.evaluate()

    print(f"\n=== {args.algo_name} on {args.task} ===")
    for key in sorted(eval_dict):
        if key.startswith("eval/") and not key.endswith("_ci95"):
            ci = eval_dict.get(f"{key}_ci95", 0.0)
            print(f"  {key[len('eval/'):]:<18s} {eval_dict[key]:10.4f}  +/- {ci:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"task": args.task, "algo": args.algo_name, **eval_dict}, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
