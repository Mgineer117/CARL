# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Construction of environments, dynamics models, and policies from parsed arguments.

Algorithm names
---------------
``--algo-name`` is ``<base>[-approx|-exact]``.  **Learned dynamics are the default**: instead of
the analytic ``f`` and ``B``, a :class:`DynamicLearner` is fitted first and the contraction
conditions are imposed on that learned model.  This is the unknown-dynamics setting, and it is the
one worth reporting -- handing a method the true model is the easy case.  ``-exact`` asks for the
analytic oracle; ``-approx`` is the default said out loud.

``ALGORITHMS`` below is the single place a name is interpreted; adding an algorithm means adding
one entry, not another ``startswith`` branch in three files.
"""

from envs import CarEnv, NeuralLanderEnv, PvtolEnv, QuadRotorEnv

ENVIRONMENTS = {
    "car": CarEnv,
    "pvtol": PvtolEnv,
    "quadrotor": QuadRotorEnv,
    "neurallander": NeuralLanderEnv,
}

#: ``base name -> (trainer kind, needs a reference preview)``.  Both algorithms here interact with
#: the environment, so ``"online"`` is currently the only kind; the field is kept so the trainer
#: choice stays data rather than a branch.
ALGORITHMS = {
    "carl": ("online", True),
    "ppo": ("online", True),
}


APPROX_SUFFIX = "-approx"
EXACT_SUFFIX = "-exact"


def parse_algo_name(algo_name: str) -> tuple[str, bool]:
    """Split an algorithm name into ``(base, uses_learned_dynamics)``.

    **Learned dynamics are the default.**  The setting worth reporting is the one where the model
    is not handed to the method, so a bare ``carl`` fits ``f`` and ``B`` from sampled transitions.
    The suffixes make the choice explicit either way:

    ``carl``          learned dynamics (the default)
    ``carl-approx``   learned dynamics, said out loud -- kept so older commands still parse
    ``carl-exact``    the analytic oracle

    """
    for suffix, approx in ((EXACT_SUFFIX, False), (APPROX_SUFFIX, True)):
        if algo_name.endswith(suffix):
            base = algo_name[: -len(suffix)]
            break
    else:
        base, approx = algo_name, True

    if base not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. Expected one of "
            f"{sorted(ALGORITHMS)} with an optional '-approx' or '-exact' suffix."
        )
    return base, approx


def call_env(args):
    """Build an environment and record its dimensions on ``args``."""
    if args.task not in ENVIRONMENTS:
        raise NotImplementedError(
            f"Unknown task '{args.task}'. Expected one of {sorted(ENVIRONMENTS)}."
        )

    span, stride = resolve_ref_window(args)
    env = ENVIRONMENTS[args.task](
        sample_mode=args.sample_mode,
        reward_mode=args.reward_mode,
        n_control_per_x=args.n_control_per_x,
        ref_span=span,
        ref_stride=stride,
    )

    # Seed the environment's own generator.  `seed_all` covers python/numpy/torch, but
    # gymnasium keeps a separate `env.np_random` that is otherwise drawn from OS entropy -- and
    # every offline dataset (the contraction buffer, the dynamics fit) is sampled
    # from it, so without this the same --seed gives different data on every run.
    env.reset(seed=int(getattr(args, "seed", 0)))

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len
    # The resolved values, for logging and for get_policy.  `ref_horizon` is derived from the
    # other two, so read it back off the environment rather than trusting the request.
    args.ref_span, args.ref_stride = env.ref_span, env.ref_stride
    args.ref_horizon = env.ref_horizon
    return env


def resolve_ref_window(args) -> tuple[int | None, int]:
    """``(span, stride)`` for the requested algorithm.  ``span=None`` means the full episode.

    CARL and PPO see the **whole reference trajectory** by default, subsampled every
    ``--ref-stride`` steps; ``--ref-span`` shortens the window for ablations.
    """
    base, _ = parse_algo_name(args.algo_name)
    _, uses_preview = ALGORITHMS[base]
    if not uses_preview:
        return 1, 1
    span = None if args.ref_span is None else max(1, int(args.ref_span))
    return span, max(1, int(args.ref_stride))


def get_dynamics(env, args, logger, writer):
    """Return ``(get_f_and_B, init_epochs)``: analytic dynamics, or a model fitted on rollouts."""
    _, approx = parse_algo_name(args.algo_name)
    if not approx:
        return env.get_f_and_B, 0

    from policy.layers.dynamics import DynamicLearner
    from trainer.offline_trainer import ComponentTrainer

    print("[INFO] Learning a dynamics approximator.")
    model = DynamicLearner(
        x_dim=env.num_dim_x,
        action_dim=args.action_dim,
        hidden_dim=args.dynamics_dim,
        dynamics_lr=args.dynamics_lr,
        drop_out=0.0,  # held-out validation now does the regularising; see ComponentTrainer
        nupdates=args.dynamics_epochs,
        device=args.device,
    )
    ComponentTrainer(
        env=env,
        module=model,
        data=env.sample_dynamics_data(args.dynamics_buffer_size),
        logger=logger,
        writer=writer,
        epochs=args.dynamics_epochs,
    ).train()

    model.eval()
    return model, args.dynamics_epochs


def get_policy(env, args, get_f_and_B):
    """Build the controller named by ``--algo-name``."""
    base, _ = parse_algo_name(args.algo_name)
    ref_horizon = env.ref_horizon

    from policy.layers.actors import RecurrentActor, RecurrentCritic
    from policy.layers.metrics import MetricNetwork

    actor = RecurrentActor(
        x_dim=env.num_dim_x,
        action_dim=args.action_dim,
        ref_dim=args.ref_dim,
        hidden_dim=args.actor_dim,
        angle_mask=env.angle_mask,
        init_logstd=args.init_logstd,
        x_min=env.X_MIN,
        x_max=env.X_MAX,
        pos_dimension=env.pos_dimension,
    )
    critic = RecurrentCritic(
        x_dim=env.num_dim_x,
        ref_dim=args.ref_dim,
        hidden_dim=args.critic_dim,
        angle_mask=env.angle_mask,
        x_min=env.X_MIN,
        x_max=env.X_MAX,
        pos_dimension=env.pos_dimension,
    )

    # The reward is defined in PPO and shared verbatim; CARL supplies only the metric.
    ppo_kwargs = dict(
        angle_mask=env.angle_mask,
        tracking_scaler=env.tracking_scaler,
        control_scaler=env.control_scaler,
        reward_mode=args.reward_mode,
        reward_form=args.reward_form,
        rl_lr=args.rl_lr,
        num_minibatch=args.num_minibatch,
        minibatch_size=args.minibatch_size,
        K=args.k_epochs,
        eps_clip=args.eps_clip,
        entropy_scaler=args.entropy_scaler,
        value_scaler=args.value_scaler,
        target_kl=args.target_kl,
        gamma=args.gamma,
        gae=args.gae,
    )

    if base == "ppo":
        from policy.ppo import PPO

        return PPO(
            x_dim=env.num_dim_x,
            ref_horizon=ref_horizon,
            actor=actor,
            critic=critic,
            device=args.device,
            **ppo_kwargs,
        )

    from policy.carl import CARL

    return CARL(
        x_dim=env.num_dim_x,
        ref_horizon=ref_horizon,
        actor=actor,
        critic=critic,
        # Stochastic: the contraction loss samples W, while the reward uses its mean.
        metric_net=MetricNetwork(
            x_dim=env.num_dim_x,
            action_dim=args.action_dim,
            hidden_dim=args.metric_dim,
            stochastic=True,
        ),
        get_f_and_B=get_f_and_B,
        data=env.sample_contraction_data(args.contraction_buffer_size),
        metric_lr=args.metric_lr,
        metric_update_interval=args.metric_update_interval,
        warmup_epochs=args.warmup_epochs,
        lbd=args.lbd,
        eps=args.eps,
        w_lb=args.w_lb,
        w_ub=args.w_ub,
        num_probes=args.num_probes,
        metric_entropy_scaler=args.metric_entropy_scaler,
        device=args.device,
        **ppo_kwargs,
    )
