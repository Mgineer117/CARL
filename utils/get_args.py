# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Command-line arguments.

Anything left at ``None`` is filled in from ``config/<task>.json`` by
:func:`utils.misc.override_args`, so per-task defaults live in one place while remaining
overridable on the command line.
"""

import argparse

import torch


def int_list(value: str) -> list[int]:
    """Parse ``"256,256"`` into ``[256, 256]``.

    ``type=list`` would apply the builtin ``list`` to the raw string and yield the *characters*
    ``['2','5','6',',',...]``, which then fails deep inside network construction.
    """
    return [int(v) for v in value.replace(" ", "").split(",") if v]


def resolve_derived(args):
    """Fill in arguments whose default is a function of another argument.

    Applied by *both* entry points below.  ``argparse`` cannot express "default to ``lbd * 0.1``",
    so these are left ``None`` by the parser and settled here -- and it has to be here rather than
    in :func:`get_args` alone, or ``training_defaults`` (which ``evaluate.py`` and the tests build
    policies from) hands ``eps=None`` straight into the contraction loss.
    """
    # Tied to the contraction rate rather than fixed, following the reference C3M implementation
    # (`epsilon = args._lambda * 0.1`): the margin the inequalities must clear should scale with
    # the rate they are being asked to certify.
    if getattr(args, "eps", None) is None:
        args.eps = args.lbd * 0.1
    return args


def get_args():
    """Parse the command line and attach the resolved torch device."""
    args = resolve_derived(build_parser().parse_args())
    args.device = select_device(args.gpu_idx, requested=args.device)
    return args


def training_defaults() -> dict:
    """Every training argument at its default value.

    ``evaluate.py`` starts from this rather than restating the defaults, so the two entry points
    cannot drift apart when a flag is renamed or added.
    """
    return vars(resolve_derived(build_parser().parse_args([])))


def build_parser() -> argparse.ArgumentParser:
    """The training argument parser.  Shared with ``evaluate.py`` via :func:`training_defaults`."""
    p = argparse.ArgumentParser(description="CARL: contraction-metric-guided RL for path tracking.")

    # --- experiment ---------------------------------------------------------------
    p.add_argument(
        "--task",
        type=str,
        default="car",
        help="car | pvtol | quadrotor | neurallander",
    )
    p.add_argument(
        "--algo-name",
        type=str,
        default="carl",
        help="carl | ppo. Dynamics are LEARNED by default (the "
        "unknown-dynamics setting); append '-exact' for the analytic oracle, or '-approx' to say "
        "the default out loud.",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed.")
    p.add_argument("--num-runs", type=int, default=5, help="Independent seeds per experiment.")
    p.add_argument("--gpu-idx", type=int, default=0, help="CUDA device index, when one is present.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device: cuda | cuda:N | mps | cpu. Auto-detected when omitted.",
    )

    # --- logging ------------------------------------------------------------------
    p.add_argument("--project", type=str, default="carl", help="WandB project.")
    p.add_argument("--logdir", type=str, default="log/train_log", help="Logging root.")
    p.add_argument("--group", type=str, default=None, help="Experiment group (multi-seed folder).")
    p.add_argument("--name", type=str, default=None, help="Run name within the group.")
    p.add_argument("--log-interval", type=int, default=50, help="Evaluations over the whole run.")
    p.add_argument(
        "--eval-num", type=int, default=10, help="Reference trajectories per evaluation."
    )
    p.add_argument(
        "--eval-episodes", type=int, default=10, help="Initial conditions per reference trajectory."
    )

    # --- budgets ------------------------------------------------------------------
    p.add_argument("--timesteps", type=int, default=None, help="Environment steps (CARL, PPO).")
    p.add_argument(
        "--dynamics-epochs", type=int, default=50_000, help="Updates for the dynamics model."
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=1000,
        help="Metric-only updates before the controller is trained.",
    )

    # --- offline datasets ---------------------------------------------------------
    p.add_argument(
        "--contraction-buffer-size",
        type=int,
        default=200_000,
        help="States for the contraction loss."
    )
    p.add_argument(
        "--dynamics-buffer-size", type=int, default=100_000, help="Samples for the dynamics fit."
    )
    p.add_argument(
        "--sample-mode",
        type=str,
        default="Uniform",
        help="Uniform | Gaussian | rollout, for the dynamics dataset.",
    )
    p.add_argument(
        "--n-control-per-x",
        type=int,
        default=1,
        help="Controls paired with each sampled state in the dynamics dataset.",
    )

    # --- architecture -------------------------------------------------------------
    p.add_argument(
        "--ref-span",
        type=int,
        default=None,
        help="How far ahead the reference preview reaches, in environment steps (CARL, PPO). "
        "Defaults to the full episode length; set a smaller value to ablate it.",
    )
    p.add_argument(
        "--ref-stride",
        type=int,
        default=2,
        help="Spacing between the previewed waypoints, so the actor sees ceil(span/stride) of "
        "them. Above 1 because the preview is not free twice over: it costs the GRU pass on every "
        "environment step, and -- since the window is encoded relative to x -- the Jacobian behind "
        "the contraction gain is backpropagated through the same steps.",
    )
    p.add_argument(
        "--ref-dim",
        type=int,
        default=32,
        help="Width of the GRU context z -- the recurrent capacity, shared by actor and critic. "
        "Reference windows on these benchmarks are very low-dimensional (PCA: 4-12 components "
        "carry 99%% of the variance of a 800-2000 number window), so 32 is already generous; it "
        "is in the sweep space if you want it tuned.",
    )
    p.add_argument(
        "--actor-dim",
        type=int_list,
        default=[128],
        help="Hidden widths of the actor's feedback weight generators (w1, w2). NOT the "
        "recurrent width -- that is --ref-dim.",
    )
    p.add_argument("--critic-dim", type=int_list, default=[256, 256], help="Critic hidden widths.")
    p.add_argument(
        "--metric-dim", type=int_list, default=[128, 128], help="Metric-network hidden widths."
    )
    p.add_argument(
        "--dynamics-dim", type=int_list, default=[256, 256], help="Dynamics-model hidden widths."
    )

    # --- optimisation -------------------------------------------------------------
    p.add_argument(
        "--rl-lr",
        type=float,
        default=3e-5,
        help="Learning rate for the RL networks -- actor and critic alike (CARL, PPO).",
    )
    p.add_argument("--metric-lr", type=float, default=3e-4, help="Metric learning rate.")
    p.add_argument("--dynamics-lr", type=float, default=1e-3, help="Dynamics-model learning rate.")
    p.add_argument("--num-minibatch", type=int, default=4, help="Minibatches per epoch.")
    p.add_argument("--minibatch-size", type=int, default=2048, help="Transitions per minibatch.")
    p.add_argument("--k-epochs", type=int, default=5, help="PPO epochs per collected batch.")
    p.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping range.")
    p.add_argument("--target-kl", type=float, default=0.01, help="PPO KL early-stopping threshold.")
    p.add_argument(
        "--entropy-scaler", type=float, default=1e-3, help="Policy entropy bonus weight (c1)."
    )
    p.add_argument(
        "--metric-entropy-scaler",
        type=float,
        default=0.01,
        help="Entropy bonus on the contraction metric generator (CMG). The generator is a "
        "diagonal Gaussian over the metric; rewarding its entropy keeps that distribution from "
        "collapsing to a point estimate, so the contraction conditions stay enforced over a "
        "neighbourhood of metrics rather than at a single one. Zero recovers the previous "
        "behaviour, where the generator's entropy entered no loss at all.",
    )
    p.add_argument(
        "--value-scaler",
        type=float,
        default=0.5,
        help="Weight of the value loss in PPO's joint objective (c2).",
    )
    p.add_argument(
        "--init-logstd",
        type=float,
        default=0.0,
        help="Initial log std of the Gaussian actor.  Exploration must be on the scale of the "
        "*correction* the controller needs, which is a fraction of the actuator range.",
    )
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    p.add_argument("--gae", type=float, default=0.95, help="GAE trace decay.")

    # --- contraction metric -------------------------------------------------------
    p.add_argument("--lbd", type=float, default=0.5, help="Desired contraction rate.")
    p.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Margin on the contraction inequalities. Defaults to lbd * 0.1, as in the reference "
        "C3M implementation; the previous fixed 0.1 was twice that at the default lbd.",
    )
    p.add_argument(
        "--num-probes",
        type=int,
        default=1024,
        help="Random directions used to test each matrix inequality. One direction per sample "
        "(the old behaviour) almost never detects a mild violation: for random unit z, "
        "E[z'Az] is the mean eigenvalue, so a single negative eigenvalue among several "
        "positive ones stays invisible.",
    )
    p.add_argument("--w-lb", type=float, default=1e-1, help="Metric lower bound.")
    p.add_argument("--w-ub", type=float, default=10.0, help="Metric upper bound.")
    p.add_argument(
        "--metric-update-interval",
        type=int,
        default=3,
        help="Policy updates between metric updates (CARL).",
    )

    # --- environment --------------------------------------------------------------
    p.add_argument(
        "--reward-mode",
        type=str,
        default="inverse",
        help="default | inverse, for the environment reward shaping.",
    )
    p.add_argument(
        "--reward-form",
        type=str,
        default="potential",
        help="How the potential becomes a reward, for CARL and PPO alike. 'potential' is "
        "Ng-Harada-Russell shaping, gamma*Phi(s_{t+1}) - Phi(s_t), which provably leaves the "
        "optimal policy unchanged for any bounded Phi. 'next' is the unprotected direct form "
        "Phi(s_{t+1}). CARL and PPO share this and every other reward setting; they differ only in "
        "the metric Phi is measured in (learned M(x) versus the identity).",
    )

    return p


def select_device(gpu_idx: int = 0, requested: str | None = None, verbose: bool = True):
    """Pick a torch device: CUDA if present, else Apple-Silicon MPS, else the CPU.

    MPS runs everything this repo needs, including the reverse-over-reverse pass behind the
    contraction loss, and agrees with the CPU to float32 precision.  Two caveats, both handled:
    ``linalg.eigvalsh`` is unimplemented there, so the eigenvalue diagnostics are computed on the
    CPU; and ``linalg.svd`` (used only to build the annihilator of a *learned* ``B``) silently
    falls back to the CPU, which costs a transfer per call but is correct.
    """
    if requested is not None:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}" if gpu_idx is not None else "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        label = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        label = "mps (Apple Silicon)"
    else:
        label = "cpu"

    if verbose:
        print("=" * 92)
        print(f"Device set to : {label}")
        print("=" * 92)
    return device
