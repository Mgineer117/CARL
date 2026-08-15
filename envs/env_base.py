# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Reference-tracking environments for control-affine systems."""

from abc import ABC, abstractmethod
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from envs.angles import tracking_error, wrap_angles


class BaseEnv(gym.Env, ABC):
    """Tracking task for a control-affine system ``xdot = f(x) + B(x) u``.

    **Observation.**  The agent sees its own state and a *preview window* of the reference
    trajectory::

        [  x_t  |  xref[t : t+H]  |  uref[t]  ]

    There is deliberately no time index: the task is to track the reference wherever you are on it,
    so the controller should be a function of the geometry, not of the clock.  Two identical
    (state, reference-ahead) situations get the same control whether they occur early or late.

    Three numbers describe that window, and only the first two are free:

    ``ref_span``     how far ahead it reaches, in environment steps (``None`` -> the full episode).
    ``ref_stride``   the spacing between the waypoints sampled from it.
    ``ref_horizon``  the resulting number of waypoints, ``ceil(ref_span / ref_stride)``.  This is
                     the ``H`` above and in :meth:`policy.base.BasePolicy.trim_state`; it is what
                     sizes the observation, so it is derived, never set directly.

    Stride is what makes a full-horizon preview affordable.  The recurrent encoder that consumes
    the window is summarised by its final hidden state, and that summary has a finite effective
    memory -- measurably a handful of steps, far short of the 200-step car episode.  Packing the
    span densely therefore buys no extra information, only cost: at ``stride=1`` the far end of
    the window is the only part that survives into ``z``, while the near end, which is the part
    that actually matters, is forgotten.  Spreading the same span over fewer waypoints puts the
    whole horizon inside the encoder's reach and cuts the recurrence, the observation width and
    the activation memory by the stride.

    ``ref_span=1`` is the classical single-point formulation, with no preview at all.  A window
    running past the end of the trajectory is **padded by repeating its final element**, so its
    width -- and hence the observation size -- is constant for every ``t``.

    Only the *current* reference control is carried, not a window of them: no network encodes
    ``uref`` -- neither the actor's GRU nor the critic's -- and the single value ``uref[t]`` is
    needed only as the feedforward term of ``u = uref[t] + feedback``.

    **Action.**  The action is the *full* control: the plant is driven by ``clip(u)``, with no
    reference control added on this side.  A controller sitting on the reference emits
    ``u = uref[t]``.

    **Reward.**  ``0.5 * tracking + 0.5 * control``, penalising ``||x_t - xref[t]||`` and ``||u||``
    with weights ``q`` and ``r``.  With ``reward_mode="inverse"`` each term is passed through
    ``1 / (1 + |.|)``, making the reward bounded in ``(0, 1]`` so that staying inside the state
    bounds is itself rewarded.

    **Episode length is constant** at ``T = round(time_bound / dt)``.  The reference trajectory is
    always generated for the full horizon (re-drawn if it would leave the state bounds), so the
    rollout buffers have an exactly predictable size.  Only the *agent* can end an episode early,
    by driving the tracked position outside its bounds.

    Subclasses provide the dynamics (:meth:`_f_logic`, :meth:`_B_logic`, and optionally
    :meth:`_B_null_logic`), the bounds and integration constants through ``env_config``, and the
    reference-control profile through :attr:`REF_FREQS`, :attr:`REF_WEIGHT_SCALE`,
    :attr:`REF_CONTROL_MASK` and :meth:`nominal_control`.
    """

    #: Frequencies (in cycles per ``time_bound``) mixed into the reference control.  Empty means a
    #: constant reference control, i.e. the reference stays at an equilibrium.
    REF_FREQS: tuple = tuple(range(1, 11))
    #: Amplitude of the sinusoidal part of the reference control.
    REF_WEIGHT_SCALE: float = 1.0
    #: Optional per-channel mask restricting which control channels are excited.
    REF_CONTROL_MASK: np.ndarray | None = None
    #: How many times to redraw a reference trajectory that leaves the state bounds.
    REF_MAX_ATTEMPTS: int = 20
    #: Indices of states that live on a circle.  These *wrap* at the bounds instead of clipping,
    #: and their tracking error is the shortest angular distance; see :mod:`envs.angles`.
    ANGLE_DIMS: tuple = ()

    def __init__(self, env_config: dict, ref_span: int | None = None, ref_stride: int = 1):
        super().__init__()

        # `env_config` is a module-level dict in each subclass, so take a copy: two environments
        # (the training one and the evaluation one) must not share mutable state.
        cfg = dict(env_config)

        # --- bounds ---------------------------------------------------------------------
        self.X_MIN, self.X_MAX = cfg["x_min"].flatten(), cfg["x_max"].flatten()
        self.UREF_MIN, self.UREF_MAX = cfg["uref_min"].flatten(), cfg["uref_max"].flatten()
        self.XREF_INIT_MIN = np.asarray(cfg["xref_init_min"]).flatten()
        self.XREF_INIT_MAX = np.asarray(cfg["xref_init_max"]).flatten()
        self.XE_INIT_MIN = np.asarray(cfg["xe_init_min"]).flatten()
        self.XE_INIT_MAX = np.asarray(cfg["xe_init_max"]).flatten()
        self.XE_MIN = np.asarray(cfg["xe_min"]).flatten()
        self.XE_MAX = np.asarray(cfg["xe_max"]).flatten()

        # --- dimensions and integration -------------------------------------------------
        self.num_dim_x = cfg["num_dim_x"]
        self.num_dim_control = cfg["num_dim_control"]
        self.pos_dimension = cfg["pos_dimension"]

        self.dt = cfg["dt"]
        self.time_bound = cfg["time_bound"]
        # round(), not int(): time_bound / dt is typically a hair under the intended integer
        # (6.0 / 0.03 == 199.999...), and int() would silently drop a step.
        self.max_episode_len = round(self.time_bound / self.dt)
        self.episode_len = self.max_episode_len
        self.t = self.dt * np.arange(self.max_episode_len)

        # Preview window: `ref_span` steps ahead, sampled every `ref_stride`.  None means "the
        # whole episode".  A stride wider than the span still yields the single current point.
        self.ref_span = self.max_episode_len if ref_span is None else max(1, int(ref_span))
        self.ref_stride = max(1, min(int(ref_stride), self.ref_span))
        self._window_offsets = np.arange(0, self.ref_span, self.ref_stride)
        self.ref_horizon = len(self._window_offsets)  # == H, derived

        # Boolean form of ANGLE_DIMS, handed to the controllers so both sides of the observation
        # contract agree on which states are angles.
        self.angle_mask = np.zeros(self.num_dim_x, dtype=bool)
        self.angle_mask[list(self.ANGLE_DIMS)] = True

        # --- reward and sampling --------------------------------------------------------
        self.tracking_scaler = cfg["q"]
        self.control_scaler = cfg["r"]
        self.reward_mode = cfg["reward_mode"]
        self.sample_mode = cfg["sample_mode"]
        self.n_control_per_x = cfg["n_control_per_x"]

        self.use_learned_dynamics = False

        # --- gymnasium spaces -----------------------------------------------------------
        H = self.ref_horizon
        self.observation_space = spaces.Box(
            low=np.concatenate((self.X_MIN, np.tile(self.X_MIN, H), self.UREF_MIN)).astype(
                np.float32
            ),
            high=np.concatenate((self.X_MAX, np.tile(self.X_MAX, H), self.UREF_MAX)).astype(
                np.float32
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self.UREF_MIN.astype(np.float32),
            high=self.UREF_MAX.astype(np.float32),
            dtype=np.float32,
        )

        self.reset()

    # ================================================================== gym API

    def reset(self, seed=None, options: dict | None = None):
        """Start a new episode.

        ``options={"replace_x_0": True}`` keeps the current reference trajectory and only redraws
        the initial state, which is how the evaluator measures several initial conditions against
        one reference.
        """
        super().reset(seed=seed)
        self.time_steps = 0

        if options is not None and options.get("replace_x_0", False):
            _, xe_0, _ = self.define_initial_state()
            self.x_0 = np.clip(
                wrap_angles(self.xref[0] + xe_0, self.angle_mask), self.X_MIN, self.X_MAX
            )
        else:
            self.x_0, self.xref, self.uref = self.system_reset()

        # Guard against a degenerate normaliser: the evaluator divides by this.
        self.init_tracking_error = max(float(np.linalg.norm(self.tracking_error(self.x_0))), 1e-8)
        self.x_t = self.x_0.copy()

        return self._observe(), {"x": self.x_t}

    def step(self, action):
        """Apply ``action`` -- the *full* control -- for one ``dt``.

        The controllers emit ``u = uref[t] + feedback`` themselves, so the reference control is
        **not** added again here; doing so would double-count it.  Clipping stays on this side of
        the interface, and the raw (unclipped) action is what the rollout buffer stores, so the
        log-probability the policy gradient uses always refers to the quantity that was actually
        sampled.
        """
        t = self.time_steps
        u = np.clip(np.asarray(action).flatten(), self.UREF_MIN, self.UREF_MAX)

        # Reward for being at x_t while the reference is at xref[t], having spent control u.
        reward, infos = self.get_rewards(u)

        self.x_t, left_bounds = self._integrate(self.x_t, u)
        self.time_steps = t + 1

        terminated = bool(left_bounds)
        truncated = self.time_steps >= self.episode_len

        # The tracking error reported here is measured at the state this call *returns*
        # (``x_{t+1}``), not at the one the reward was computed for (``x_t``).  The reward is
        # rightly a function of where you were when you acted, but the info dict must be
        # self-consistent: reporting the pre-step error alongside the post-step state made the
        # evaluator's error trace `[e_0, e_0, e_1, ..., e_{T-1}]` -- first point duplicated, last
        # one dropped -- which biases every mAUC by about one `dt`.
        next_error = float(np.linalg.norm(self.tracking_error(self.x_t)))

        return (
            self._observe(),
            reward,
            terminated,
            truncated,
            {
                "x": self.x_t,
                "tracking_error": next_error,
                "relative_tracking_error": next_error / self.init_tracking_error,
                "control_effort": infos["control_effort"],
            },
        )

    def render(self):
        """No on-screen rendering; the evaluator plots trajectories instead."""

    # ================================================================== observation

    def _observe(self) -> np.ndarray:
        """Build the flat observation described in the class docstring."""
        xref_win, uref_win = self.reference_window(self.time_steps)
        return np.concatenate((self.x_t, xref_win.ravel(), uref_win[0])).astype(np.float32)

    def reference_window(self, t: int):
        """The waypoints at ``t, t + stride, ...`` covering the next ``ref_span`` steps.

        ``xref`` has ``T + 1`` entries and ``uref`` has ``T``, so each index is clamped to its own
        last valid position: a window that runs off the end repeats the final reference point and
        the final reference control rather than wrapping round or shrinking.
        """
        offsets = t + self._window_offsets
        xref_win = self.xref[np.minimum(offsets, len(self.xref) - 1)]
        uref_win = self.uref[np.minimum(offsets, len(self.uref) - 1)]
        return xref_win, uref_win

    # ================================================================== reward

    def tracking_error(self, x: np.ndarray, t: int | None = None) -> np.ndarray:
        """``x`` minus the reference at step ``t``, with circular states wrapped."""
        t = self.time_steps if t is None else t
        return tracking_error(x, self.xref[t], self.angle_mask)

    def get_rewards(self, u: np.ndarray):
        """Reward for the current state/reference pair under control ``u``."""
        error = float(np.linalg.norm(self.tracking_error(self.x_t)))
        control_effort = float(np.linalg.norm(u))

        tracking_reward = -self.tracking_scaler * error
        control_reward = -self.control_scaler * control_effort

        if self.reward_mode == "inverse":
            tracking_reward = 1.0 / (1.0 + abs(tracking_reward))
            control_reward = 1.0 / (1.0 + abs(control_reward))

        # `control_scaler = 0` means *no control term*, not a term that happens to be constant.
        # Under "inverse" a zero scaler still yields 1/(1+0) = 1, so keeping the 0.5/0.5 split
        # would add a fixed 0.5 to every reward -- halving the dispersion of the only part that
        # carries signal, which is exactly what the policy gradient needs.
        if self.control_scaler == 0:
            reward = tracking_reward
        else:
            reward = 0.5 * tracking_reward + 0.5 * control_reward
        return reward, {"tracking_error": error, "control_effort": control_effort}

    # ================================================================== dynamics

    @staticmethod
    def _zeros(x, shape, lib):
        """Zeros matching ``x``'s dtype and device.

        The dynamics are evaluated on GPU tensors by the contraction loss, so a bare
        ``torch.zeros`` (which lands on the CPU) would make every subclass fail there.
        """
        if lib is torch:
            return torch.zeros(shape, dtype=x.dtype, device=x.device)
        return np.zeros(shape)

    @abstractmethod
    def _f_logic(self, x, lib):
        """``f(x)`` for a batch ``x`` of shape ``(n, num_dim_x)``, using ``lib`` (``torch`` or ``np``)."""

    @abstractmethod
    def _B_logic(self, x, lib):
        """``B(x)`` of shape ``(n, num_dim_x, num_dim_control)``."""

    def _B_null_logic(self, x, n, lib):
        """Default annihilator of ``B``: valid whenever ``B`` only drives the *last* states.

        Returns ``[[I], [0]]`` of shape ``(n, num_dim_x, num_dim_x - num_dim_control)``, which
        satisfies ``B^T Bbot = 0`` exactly when the non-zero rows of ``B`` are the trailing
        ``num_dim_control`` ones.  A system where that does not hold overrides this.
        """
        eye_dims = self.num_dim_x - self.num_dim_control
        if lib is torch:
            Bbot = torch.cat(
                (
                    torch.eye(eye_dims, dtype=x.dtype, device=x.device),
                    torch.zeros(self.num_dim_control, eye_dims, dtype=x.dtype, device=x.device),
                ),
                dim=0,
            )
            return Bbot.expand(n, -1, -1)

        Bbot = np.concatenate(
            (np.eye(eye_dims), np.zeros((self.num_dim_control, eye_dims))), axis=0
        )
        return np.repeat(Bbot[np.newaxis], n, axis=0)

    def f_func(self, x):
        """Drift ``f(x)``, batched or unbatched, for a numpy array or a torch tensor."""
        return self._dispatch(self._f_logic, x)

    def B_func(self, x):
        """Control matrix ``B(x)``, batched or unbatched."""
        return self._dispatch(self._B_logic, x)

    def B_null(self, x):
        """Annihilator ``Bbot(x)`` with ``B^T Bbot = 0``, batched or unbatched."""
        return self._dispatch(lambda xx, lib: self._B_null_logic(xx, xx.shape[0], lib), x)

    @staticmethod
    def _dispatch(logic, x):
        """Run ``logic`` on a batched ``x``, restoring an unbatched input's shape on the way out."""
        unbatched = x.ndim == 1
        if unbatched:
            x = x.unsqueeze(0) if isinstance(x, torch.Tensor) else x[np.newaxis, :]
        result = logic(x, torch if isinstance(x, torch.Tensor) else np)
        return result.squeeze(0) if unbatched else result

    def get_f_and_B(self, x):
        """``(f(x), B(x), Bbot(x))``, from the learned dynamics model if one was installed."""
        if self.use_learned_dynamics:
            unbatched = x.ndim == 1
            with torch.no_grad():
                out = self.learned_dynamics_model(x)
            return tuple((t.squeeze(0) if unbatched else t).cpu().numpy() for t in out)
        return self.f_func(x), self.B_func(x), self.B_null(x)

    def replace_dynamics(self, dynamics_model: nn.Module):
        """Make the *simulator* itself use a learned dynamics model (off by default)."""
        print("[INFO] The environment is now using learned dynamics for transition.")
        self.learned_dynamics_model = deepcopy(dynamics_model).cpu()
        self.learned_dynamics_model.device = torch.device("cpu")
        self.use_learned_dynamics = True

    def get_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """``xdot = f(x) + B(x) u``, batched or unbatched."""
        f_x, B_x, _ = self.get_f_and_B(x)
        return f_x + np.matmul(B_x, u[..., np.newaxis])[..., 0]

    def _integrate(self, x: np.ndarray, u: np.ndarray):
        """One explicit-Euler step, clipped to the state bounds.

        Returns ``(next_x, left_bounds)`` where ``left_bounds`` says whether the *tracked* position
        coordinates hit their limits -- the condition that ends an episode.
        """
        next_x = x + self.dt * self.get_dynamics(x, u)

        # Circular states wrap; clipping them would freeze that degree of freedom at the cut.
        # After wrapping they already lie inside their bounds, so the clip below is a no-op there.
        next_x = wrap_angles(next_x, self.angle_mask)

        pos = next_x[: self.pos_dimension]
        left_bounds = bool(
            np.any(pos <= self.X_MIN[: self.pos_dimension])
            or np.any(pos >= self.X_MAX[: self.pos_dimension])
        )
        return np.clip(next_x, self.X_MIN, self.X_MAX), left_bounds

    # ================================================================== reference

    def define_initial_state(self):
        """Draw ``(xref_0, xe_0, x_0)``: a reference anchor and a perturbed initial state."""
        xref_0 = self.XREF_INIT_MIN + self.np_random.random(self.num_dim_x) * (
            self.XREF_INIT_MAX - self.XREF_INIT_MIN
        )
        xe_0 = self.XE_INIT_MIN + self.np_random.random(self.num_dim_x) * (
            self.XE_INIT_MAX - self.XE_INIT_MIN
        )
        # The initial state has to be a valid state, so clip it -- otherwise the very first
        # observation can sit outside observation_space (e.g. the car starting below its speed
        # limit) and the reported initial tracking error would not match the simulated one.
        x_0 = wrap_angles(xref_0 + xe_0, self.angle_mask)
        return xref_0, xe_0, np.clip(x_0, self.X_MIN, self.X_MAX)

    def nominal_control(self, xref_0: np.ndarray) -> np.ndarray:
        """Constant part of the reference control -- usually the equilibrium/trim control."""
        return np.zeros(self.num_dim_control)

    def sample_reference_controls(
        self, weights: np.ndarray, t: float, xref_0: np.ndarray, add_noise: bool = False
    ) -> np.ndarray:
        """Reference control at time ``t``: trim plus a fixed mix of sinusoids."""
        uref = self.nominal_control(xref_0).astype(np.float64)
        for freq, weight in zip(self.REF_FREQS, weights):
            uref = uref + weight * np.sin(2 * np.pi * freq * t / self.time_bound)
        if add_noise:
            uref = uref + self.np_random.normal(0.0, np.abs(0.1 * uref) + 1e-12)
        return np.clip(uref, self.UREF_MIN, self.UREF_MAX)

    def _draw_reference_weights(self) -> np.ndarray:
        """Per-frequency control amplitudes, normalised so the total amplitude is scale-controlled."""
        if len(self.REF_FREQS) == 0:
            return np.zeros((0, self.num_dim_control))

        w = self.np_random.standard_normal((len(self.REF_FREQS), self.num_dim_control))
        w = self.REF_WEIGHT_SCALE * w / np.sqrt((w**2).sum(axis=0, keepdims=True))
        if self.REF_CONTROL_MASK is not None:
            w = w * self.REF_CONTROL_MASK
        return w

    def system_reset(self):
        """Draw an initial state and a full-horizon reference trajectory.

        Returns ``(x_0, xref, uref)`` with ``xref`` of length ``max_episode_len + 1`` and ``uref`` of
        length ``max_episode_len``.  A reference that would leave the state bounds is redrawn rather
        than truncated, so every episode has the same length.
        """
        for _ in range(self.REF_MAX_ATTEMPTS):
            xref_0, _, x_0 = self.define_initial_state()
            weights = self._draw_reference_weights()

            xref, uref, escaped = [xref_0], [], False
            for t in self.t:
                u_t = self.sample_reference_controls(weights, t, xref_0)
                next_xref, left_bounds = self._integrate(xref[-1], u_t)
                xref.append(next_xref)
                uref.append(u_t)
                if left_bounds:
                    escaped = True
                    break

            if not escaped:
                return x_0, np.array(xref), np.array(uref)

        # Fall back on the last (bound-clipped, hold-the-end) attempt rather than loop forever.
        while len(uref) < self.max_episode_len:
            uref.append(uref[-1])
            xref.append(xref[-1])
        return x_0, np.array(xref), np.array(uref)

    # ================================================================== offline datasets

    def sample_contraction_data(
        self, num_samples: int, num_reference_trajectories: int = 256
    ) -> "ContractionDataset":
        """States and reference preview windows for the contraction losses.

        The two halves are drawn from different distributions on purpose:

        * the **state** ``x`` is drawn **uniformly over the state box**, and deliberately *not*
          from the states the policy visits.  The contraction conditions are pointwise constraints
          that have to hold over the whole operating region, so that is where the metric is fitted;
          training it on the policy's own visitation would certify only the region the policy
          already reaches, which is exactly where a certificate is least needed.

          It is a flat uniform rather than a reference point plus a perturbation.  The latter must
          be clipped back into the box, and for a narrow coordinate the clipping dominates: on the
          car's speed, whose range ``[1, 2]`` is half the perturbation's width, **50%** of samples
          landed exactly on a bound, and on the quadrotor's attitude **24%** did -- so the metric
          was being fitted mostly to the boundary of those coordinates.
        * the **reference window** comes from a pool of genuine reference trajectories, i.e. the
          same distribution the policy is conditioned on at deployment.  The window only supplies
          the preview context for the gain ``K = du/dx``, so it need not be tied to ``x``.

        Windows are stored as ``(trajectory, start index)`` pairs rather than materialised: at the
        full episode horizon a materialised buffer would be ``num_samples x H x x_dim`` floats --
        gigabytes -- while the pool it is cut from is a few megabytes.

        The window geometry is taken from this environment rather than passed in, so the offline
        windows the metric is fitted on are cut exactly like the online ones the policy is
        conditioned on.  They condition the same gain ``K = du/dx`` that the contraction
        conditions constrain, so a mismatch -- a contiguous 20-step window offline against a
        strided 200-step one online -- would silently fit the metric to the wrong distribution.
        """
        x = self.np_random.uniform(
            self.X_MIN, self.X_MAX, size=(num_samples, self.num_dim_x)
        ).astype(np.float32)

        pool_xref, pool_uref = [], []
        for _ in range(num_reference_trajectories):
            _, xref, uref = self.system_reset()
            pool_xref.append(xref)
            pool_uref.append(uref)

        return ContractionDataset(
            x=x,
            xref_pool=np.asarray(pool_xref, dtype=np.float32),
            uref_pool=np.asarray(pool_uref, dtype=np.float32),
            offsets=self._window_offsets,
            rng=self.np_random,
        )

    def sample_dynamics_data(self, num_samples: int) -> "TransitionDataset":
        """``(x, u, xdot)`` triples for fitting the dynamics approximator.

        ``sample_mode`` picks how the state/control pairs are drawn: ``"Uniform"`` or ``"Gaussian"``
        over the boxes, or anything else for on-policy-like rollouts of the reference controller.
        ``n_control_per_x`` reuses each state with several controls, which identifies ``B(x)``
        without needing more distinct states.
        """
        if self.sample_mode in ("Uniform", "Gaussian"):
            per_control = int(np.ceil(num_samples / self.n_control_per_x))

            if self.sample_mode == "Uniform":
                x = self.np_random.uniform(self.X_MIN, self.X_MAX, (per_control, self.num_dim_x))
                u = self.np_random.uniform(
                    self.UREF_MIN, self.UREF_MAX, (per_control, self.num_dim_control)
                )
            else:  # 3-sigma of the Gaussian covers the box
                x = self.np_random.normal(
                    (self.X_MAX + self.X_MIN) / 2,
                    (self.X_MAX - self.X_MIN) / 6,
                    (per_control, self.num_dim_x),
                )
                u = self.np_random.normal(
                    (self.UREF_MAX + self.UREF_MIN) / 2,
                    (self.UREF_MAX - self.UREF_MIN) / 6,
                    (per_control, self.num_dim_control),
                )

            x = np.tile(x, (self.n_control_per_x, 1))
            u = np.concatenate(
                [u[self.np_random.permutation(len(u))] for _ in range(self.n_control_per_x)]
            )
            x_dot = self.get_dynamics(x, u)
        else:
            x, u, x_dot = self._rollout_dynamics_data(num_samples)

        return TransitionDataset(
            x=x[:num_samples].astype(np.float32),
            u=u[:num_samples].astype(np.float32),
            x_dot=x_dot[:num_samples].astype(np.float32),
            rng=self.np_random,
        )

    def _rollout_dynamics_data(self, num_samples: int):
        """Collect ``(x, u, xdot)`` by running the reference controller with added noise."""
        xs, us, xdots = [], [], []
        while len(xs) < num_samples:
            xref_0, _, x_t = self.define_initial_state()
            weights = self._draw_reference_weights()

            for t in self.t:
                u_t = self.sample_reference_controls(weights, t, xref_0, add_noise=True)
                xs.append(x_t)
                us.append(u_t)
                xdots.append(self.get_dynamics(x_t, u_t))
                x_t, left_bounds = self._integrate(x_t, u_t)
                if left_bounds or len(xs) >= num_samples:
                    break

        return np.array(xs), np.array(us), np.array(xdots)


# ===================================================================== offline datasets


class TransitionDataset:
    """``(x, u, xdot)`` triples for fitting the dynamics approximator.

    Split into a training and a validation half on construction.  :meth:`sample` draws only from
    the training part, so the held-out part stays a genuine estimate of generalisation and can be
    used to decide when to stop -- fitting the dynamics for a fixed number of epochs otherwise has
    no way to tell "converged" from "started memorising the sample".

    Args:
        x / u / x_dot: the transitions.
        rng: the environment's seeded generator, so the split and the batches are reproducible.
        val_fraction: share held out for validation.
    """

    def __init__(self, x: np.ndarray, u: np.ndarray, x_dot: np.ndarray, rng, val_fraction=0.2):
        self.rng = rng

        n = len(x)
        # Permute before splitting: the rollout sampler emits transitions in trajectory order, so
        # a contiguous tail would be a handful of episodes rather than a sample of the state space.
        order = rng.permutation(n)
        n_val = int(round(val_fraction * n))
        val_idx, train_idx = order[:n_val], order[n_val:]

        self.x, self.u, self.x_dot = x[train_idx], u[train_idx], x_dot[train_idx]
        self.val = {"x": x[val_idx], "u": u[val_idx], "x_dot": x_dot[val_idx]}

    def __len__(self) -> int:
        return len(self.x)

    def sample(self, batch_size: int) -> dict:
        """Draw a minibatch of *training* transitions without replacement."""
        idx = self.rng.choice(len(self), size=min(batch_size, len(self)), replace=False)
        return {"x": self.x[idx], "u": self.u[idx], "x_dot": self.x_dot[idx]}

    def validation(self, batch_size: int | None = None) -> dict:
        """The held-out transitions, optionally subsampled to bound the cost of one check."""
        if batch_size is None or batch_size >= len(self.val["x"]):
            return self.val
        idx = self.rng.choice(len(self.val["x"]), size=batch_size, replace=False)
        return {k: v[idx] for k, v in self.val.items()}


class ContractionDataset:
    """States paired with reference preview windows, for the contraction losses.

    The windows are cut lazily out of ``xref_pool`` / ``uref_pool``: a batch stores only which
    trajectory and which start index it came from, so memory is ``pool_size x T`` rather than
    ``num_samples x H``.  A window running past the end of a trajectory holds its final point,
    matching ``BaseEnv._observe``.

    Args:
        x: ``(N, x_dim)`` states covering the operating region.
        xref_pool: ``(P, T + 1, x_dim)`` reference trajectories.
        uref_pool: ``(P, T, u_dim)`` their reference controls.
        offsets: the window's step offsets, ``BaseEnv._window_offsets``.  Shared rather than
            recomputed so offline and online windows cannot drift apart.
        rng: the environment's seeded generator, so datasets are reproducible with the run.
    """

    def __init__(self, x, xref_pool, uref_pool, offsets, rng):
        self.x = x
        self.xref_pool = xref_pool
        self.uref_pool = uref_pool
        self.offsets = np.asarray(offsets)
        self.horizon = len(self.offsets)
        self.rng = rng
        self.episode_len = uref_pool.shape[1]

    def __len__(self) -> int:
        return len(self.x)

    def sample(self, batch_size: int) -> dict:
        """Return ``x`` ``(n, x_dim)``, ``xref`` ``(n, H, x_dim)``, ``uref`` ``(n, u_dim)``.

        Only the *current* reference control, matching ``BaseEnv._observe``: nothing encodes a
        window of them, and the controller needs ``uref[t]`` alone as its feedforward term.
        """
        n = min(batch_size, len(self))
        idx = self.rng.choice(len(self), size=n, replace=False)
        traj = self.rng.integers(len(self.xref_pool), size=n)
        start = self.rng.integers(self.episode_len, size=n)

        offsets = start[:, None] + self.offsets[None, :]
        return {
            "x": self.x[idx],
            "xref": self.xref_pool[traj[:, None], np.minimum(offsets, self.episode_len)],
            "uref": self.uref_pool[traj, np.minimum(start, self.episode_len - 1)],
        }
