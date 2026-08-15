# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Entry point.

    python main.py --task car --algo-name carl
    python main.py --task neurallander --algo-name carl          # unknown dynamics (default)

Runs ``--num-runs`` seeds of one (task, algorithm) pair and merges their CSV logs at the end.
"""

import datetime
import random
import uuid

import torch

import wandb
from trainer.online_trainer import OnlineTrainer
from utils.get_args import get_args
from utils.misc import concat_csv_columnwise_and_delete, override_args, seed_all, setup_logger
from utils.sampler import OnlineSampler
from utils.utils import call_env, get_dynamics, get_policy


def run(args, seed: int, unique_id: str, exp_time: str):
    """One seed: build everything, train (or just evaluate), and shut the logger down."""
    seed_all(seed)

    # Both environments simulate the *true* system, in every setting.  A learned dynamics model
    # never reaches the simulator: `get_dynamics` hands it to the policy, where it is used only by
    # the contraction conditions.  So under `-approx` it is the *method* that is
    # denied the model -- the metric certifies contraction of a system it merely estimated -- while
    # the world the policy is scored in stays exact.  `BaseEnv.replace_dynamics` exists to simulate
    # the learned model instead, and is deliberately never called; mAUC is therefore comparable
    # across the known- and unknown-dynamics settings, because the plant is identical in both.
    env = call_env(args)
    eval_env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    get_f_and_B, init_epochs = get_dynamics(env, args, logger, writer)
    policy = get_policy(env, args, get_f_and_B)

    trainer_kwargs = dict(
        env=env,
        eval_env=eval_env,
        policy=policy,
        logger=logger,
        writer=writer,
        init_epochs=init_epochs,
        log_interval=args.log_interval,
        eval_num=args.eval_num,
        eval_episodes=args.eval_episodes,
        seed=seed,
    )

    sampler = OnlineSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=args.minibatch_size * args.num_minibatch,
    )
    OnlineTrainer(sampler=sampler, epochs=args.timesteps, **trainer_kwargs).train()

    wandb.finish()


def main():
    """Run every seed of one (task, algorithm) pair and merge their CSV logs."""
    torch.set_default_dtype(torch.float32)
    init_args = get_args()

    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    # Derive the per-run seeds deterministically from the base seed.
    seeds = random.Random(init_args.seed).sample(range(1, 10_001), init_args.num_runs)
    print("-" * 55)
    print(f"      Running ID   : {unique_id}")
    print(f"      Task / Algo  : {init_args.task} / {init_args.algo_name}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print("-" * 55)

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed
        run(args, seed, unique_id, exp_time)

    concat_csv_columnwise_and_delete(folder_path=args.logdir)


if __name__ == "__main__":
    main()
