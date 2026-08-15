# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Actor and critic networks.

Three pieces, in dependency order:

``BilinearFeedback``
    The feedback parameterisation ``u = W2(c) @ tanh( W1(c) @ e )``, where ``e = x - xref`` is
    the tracking error and ``c`` is a conditioning vector.  Both actors below are this head with a
    different ``c``.

``RecurrentActor``
    ``c = [x', z]`` where ``z`` is a GRU summary of the whole reference preview window and ``x'``
    is the state in reference-relative coordinates.  Used by CARL and PPO.

``RecurrentCritic``
    Value function with the same GRU encoding of the reference as ``RecurrentActor``, conditioned
    on the state and the reference *states* only (no ``uref``).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from envs.angles import tracking_error
from policy.layers.building_blocks import MLP


class _StateAware(nn.Module):
    """Mixin giving a network the environment's state geometry: which states are angles, and scale.

    **Angles.**  Without the mask the network would form ``x - xref`` as a plain subtraction, and a
    heading either side of the branch cut would read as ~2*pi of tracking error instead of ~0.

    **Scale.**  States are physical quantities: the car's position spans +-15 m while its speed
    spans 1-2 m/s.  Fed raw into ``tanh`` layers the wide ones saturate -- ``tanh(15)`` has a
    gradient of ``0`` to float precision -- so the network is deaf to exactly the coordinates that
    carry the tracking error.  Normalising by the environment's own state box puts every coordinate
    in ``[-1, 1]``.

    The bounds are constants of the task, so this is a fixed linear rescale rather than running
    statistics: reproducible, and identical between training and evaluation.  It is also invisible
    to the contraction conditions -- ``K = du/dx`` is taken through the composed function, so
    autograd applies the chain rule for the rescale automatically.
    """

    def register_state_geometry(
        self, x_dim: int, angle_mask=None, x_min=None, x_max=None, pos_dimension: int = 0
    ):
        """Record the state geometry as buffers: the circular mask and the normalisation box."""
        mask = np.zeros(x_dim, dtype=bool) if angle_mask is None else np.asarray(angle_mask, bool)
        # Buffers, so they follow the module across devices.
        self.register_buffer("angle_mask", torch.as_tensor(mask), persistent=False)
        #: How many leading coordinates are positions, and so are encoded relative to ``x``.
        self.pos_dimension = int(pos_dimension)

        if x_min is None or x_max is None:
            center, scale = np.zeros(x_dim), np.ones(x_dim)
        else:
            lo, hi = np.asarray(x_min, float).ravel(), np.asarray(x_max, float).ravel()
            center, scale = (hi + lo) / 2.0, np.maximum((hi - lo) / 2.0, 1e-6)
        self.register_buffer(
            "state_center", torch.as_tensor(center, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "state_scale", torch.as_tensor(scale, dtype=torch.float32), persistent=False
        )

    def error(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """Tracking error with circular states reduced to their shortest angular distance."""
        return tracking_error(x, xref, self.angle_mask)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map a state into ``[-1, 1]`` using the environment's state box."""
        return (x - self.state_center) / self.state_scale

    def normalize_delta(self, dx: torch.Tensor) -> torch.Tensor:
        """Scale a *difference* of states; differences are already centred, so no offset."""
        return dx / self.state_scale

    def encode_window(self, x: torch.Tensor, xref_seq: torch.Tensor) -> torch.Tensor:
        """Encode a preview window ``(n, H, x_dim)`` in coordinates centred on ``x``.

        The leading ``pos_dimension`` coordinates are *positions*, and they are the ones that
        translate: the same manoeuvre executed at one end of the arena or the other is the same
        control problem, but in absolute coordinates it is two unrelated inputs the network has to
        learn about separately.  Subtracting the current position collapses them, so the encoder
        sees the path's shape and where it lies *relative to the vehicle* -- which is what the
        controller acts on.  The remaining coordinates (headings, rates) are not translatable and
        stay absolute, normalised by the state box.

        Both halves are divided by ``state_scale``, so an offset of a full arena width is O(1)
        rather than O(10) and the ``tanh`` layers downstream stay in their responsive region.
        """
        p = self.pos_dimension
        window = self.normalize(xref_seq)
        if p:
            # Slice the scale to match: state_scale is full width, this difference is p wide.
            relative = (xref_seq[..., :p] - x[:, None, :p]) / self.state_scale[:p]
            window = torch.cat((relative, window[..., p:]), dim=-1)
        return window

    def encode_state(self, x: torch.Tensor, xref: torch.Tensor) -> torch.Tensor:
        """Encode the current state ``(n, x_dim)`` with its positions relative to the reference.

        The counterpart of :meth:`encode_window`, and for the same reason.  That method already
        removes absolute position from the *preview*; this removes it from the *state*, so that
        after both no network input carries an absolute location at all.  Position enters only ever
        as a displacement -- the reference seen from the vehicle (``xref - x``) and the vehicle
        seen from the reference (``x - xref[t]``).

        Without this the conditioning vector still pinned the controller to where in the arena it
        was: the identical tracking situation at two ends of the box produced two different inputs
        to the weight generators, so the network had to learn the same control law once per
        region.  The dynamics of these benchmarks are translation-invariant in their position
        coordinates, so that is capacity spent on a symmetry the system already has.

        The remaining coordinates (headings, rates) are not translatable and stay absolute,
        normalised by the state box.
        """
        p = self.pos_dimension
        state = self.normalize(x)
        if p:
            relative = (x[..., :p] - xref[..., :p]) / self.state_scale[:p]
            state = torch.cat((relative, state[..., p:]), dim=-1)
        return state


class BilinearFeedback(nn.Module):
    """State-dependent feedback ``u = W2(c) @ tanh( W1(c) @ e )`` on the tracking error ``e``.

    The reason for this form (rather than a plain MLP on
    the concatenated observation) is structural: ``u -> 0`` as ``e -> 0``, so the network only ever
    produces a *correction* to the reference control ``uref``, and the reference trajectory is an
    equilibrium of the closed loop by construction rather than by training.

    ``w2`` -- the generator of the layer that actually emits the action -- uses the small-gain
    actor initialisation, so an untrained controller starts at ``u ~ 0``: pure feedforward, which
    is a sane control law, and learning moves away from it.  With a generic initialisation the
    same network starts by emitting ``|u| ~ 7`` against a control bound of 3, i.e. saturated
    actuators driven by noise, which is *worse* than doing nothing and is a poor place to start
    a policy-gradient method from.

    Args:
        cond_dim: width of the conditioning vector ``c``.
        x_dim: state dimension (``e`` lives in ``R^x_dim``).
        action_dim: control dimension.
        hidden_dim: hidden widths of the two weight generators (a list, so depth is tunable).
        latent_scale: the intermediate width is ``latent_scale * x_dim``.
    """

    def __init__(
        self,
        cond_dim: int,
        x_dim: int,
        action_dim: int,
        hidden_dim=(128,),
        latent_scale: int = 3,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.action_dim = action_dim
        latent_dim = latent_scale * x_dim

        hidden_dim = list(hidden_dim)
        self.w1 = MLP(cond_dim, hidden_dim, latent_dim * x_dim, activation=nn.Tanh())
        self.w2 = MLP(
            cond_dim,
            hidden_dim,
            action_dim * latent_dim,
            activation=nn.Tanh(),
            initialization="actor",
        )

    def forward(self, cond: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Control correction for the tracking error ``e`` under conditioning vector ``cond``."""
        n = cond.shape[0]
        w1 = self.w1(cond).view(n, -1, self.x_dim)  # (n, latent_dim, x_dim)
        w2 = self.w2(cond).view(n, self.action_dim, -1)  # (n, action_dim, latent_dim)
        latent = torch.tanh(w1 @ e.unsqueeze(-1))  # (n, latent_dim, 1)
        return (w2 @ latent).squeeze(-1)  # (n, action_dim)


class GaussianHead(_StateAware):
    """Shared diagonal-Gaussian sampling / log-prob / entropy for the actors below.

    The scale is a state-independent learnable parameter, which keeps ``du/dx`` -- the gain matrix
    ``K`` that the contraction condition needs -- equal to the Jacobian of the mean.

    ``init_logstd`` starts exploration well below the control bound.  The default of 0 (unit
    standard deviation) is far too wide for a *correction* to a reference control: on these
    benchmarks the useful corrections are a fraction of the actuator range, so unit noise simply
    saturates the actuators and drowns the signal the advantage estimate is meant to carry.
    """

    LOGSTD_RANGE = (-5.0, 2.0)

    def __init__(self, action_dim: int, init_logstd: float = -1.0):
        super().__init__()
        self.action_dim = action_dim
        self.logstd = nn.Parameter(torch.full((1, action_dim), float(init_logstd)))

    def distribution(self, mu: torch.Tensor) -> Normal:
        """Diagonal Gaussian around the mean action, with the scale clamped for stability."""
        logstd = torch.clamp(self.logstd, *self.LOGSTD_RANGE).expand_as(mu)
        # validate_args=False: the clamp above already guarantees a positive, finite scale, and
        # torch's constraint check runs on every construction -- once per environment step, where
        # it costs more than the rest of the forward pass.
        return Normal(loc=mu, scale=torch.exp(logstd), validate_args=False)

    def sample(self, mu: torch.Tensor, deterministic: bool):
        """Return ``(action, info)``; every entry of ``info`` is shaped ``(n, 1)``."""
        dist = self.distribution(mu)
        a = mu if deterministic else dist.rsample()
        logprobs = self.log_prob(dist, a)
        return a, {
            "dist": dist,
            "logprobs": logprobs,
            "probs": torch.exp(logprobs),
            "entropy": self.entropy(dist),
        }

    @staticmethod
    def log_prob(dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        """Joint log-density of a whole action vector -> ``(n, 1)``."""
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)

    @staticmethod
    def entropy(dist: Normal) -> torch.Tensor:
        """Joint entropy of the action distribution -> ``(n, 1)``."""
        return dist.entropy().sum(dim=-1, keepdim=True)


class RecurrentActor(GaussianHead):
    """CARL / PPO actor: conditions on the whole reference *preview window* via a GRU.

    The observation carries ``xref[t:t+H]`` and ``uref[t]`` (see ``BaseEnv._observe``).  A GRU
    compresses that window into one context vector ``z``, which then parameterises the same
    bilinear feedback law:

        z = GRU( xref[t:t+H] - x  (positions) )
        u = uref[t] + W2(x', z) @ tanh( W1(x', z) @ (x - xref[t]) )

    where ``x'`` is the state with its own positions taken relative to ``xref[t]``.  The weight
    generators condition on ``(x', z)`` only: the reference enters *entirely* through the recurrent
    context, and the tracking error ``e`` enters only where the structure requires it.  Two
    properties of this factorisation matter:

    * **Position appears only ever as a displacement** -- the reference seen from the vehicle in
      ``z``, the vehicle seen from the reference in ``x'`` -- so no absolute location reaches any
      layer and the same manoeuvre is the same input wherever in the arena it happens.  This
      matches the systems themselves, whose dynamics are translation-invariant in position.
    * **``z`` therefore depends on ``x``**, and the closed-loop gain ``K = du/dx`` that the
      contraction loss needs picks up the path through the recurrence.  This is the honest gain --
      moving the state really does move the encoded window -- but it means the Jacobian is
      backpropagated through ``H`` GRU steps, so the preview horizon is no longer free in the
      contraction loss.  ``--ref-stride`` is what keeps ``H`` small enough for that to be cheap.

    Args:
        x_dim: state dimension.
        action_dim: control dimension.
        ref_dim: width of the GRU context ``z``.
        hidden_dim: hidden widths of the feedback weight generators.  This is *not* the recurrent
            width -- that is ``ref_dim``.
    """

    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        ref_dim: int = 64,
        hidden_dim=(128,),
        angle_mask=None,
        init_logstd: float = -1.0,
        x_min=None,
        x_max=None,
        pos_dimension: int = 0,
    ):
        super().__init__(action_dim, init_logstd)
        self.x_dim = x_dim
        self.ref_dim = ref_dim

        # xref only: the reference *controls* never enter the encoder.
        self.ref_encoder = nn.GRU(x_dim, ref_dim, batch_first=True)
        # Conditioning is (x, z): the reference reaches the weight generators only through z.
        self.feedback = BilinearFeedback(x_dim + ref_dim, x_dim, action_dim, hidden_dim)
        self.register_state_geometry(x_dim, angle_mask, x_min, x_max, pos_dimension)

    def encode_reference(self, x: torch.Tensor, xref_seq: torch.Tensor) -> torch.Tensor:
        """Summarise the reference window ``(n, H, x_dim)`` into a context vector ``(n, ref_dim)``.

        Positions are taken **relative to** ``x`` (see :meth:`_StateAware.encode_window`), so ``z``
        describes the path ahead as seen from the vehicle.

        The window is fed to the GRU **in reverse**, so the *nearest* waypoint is the last thing the
        recurrence sees.  A GRU summarised by its final hidden state is dominated by its final inputs,
        and over a 200-step window that dominance is total: measured on this benchmark, forward
        ordering gives ``xref[t]`` -- where the reference is right now -- an influence on ``z`` of
        0.000000, while the last 20 (most distant) steps carry 100%.  That is precisely backwards for
        tracking, where the near future matters most.  Reversing costs nothing and inverts it.
        """
        _, h_last = self.ref_encoder(self.encode_window(x, xref_seq).flip(1))
        return h_last.squeeze(0)

    def mean(
        self, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """``u = uref + W2([x,z]) tanh(W1([x,z]) e)``; ``K = du/dx`` differentiates this.

        ``uref`` enters as a pure feedforward offset.  It does not depend on ``x``, so it leaves
        the closed-loop gain untouched, and the small-gain initialisation of ``w2`` now means an
        untrained controller emits ``u ~ uref`` -- the reference control itself, which is a sane
        starting policy rather than an arbitrary one.

        The state is encoded with its positions taken relative to ``xref`` (see
        :meth:`_StateAware.encode_state`), so no absolute location reaches the weight generators.
        """
        cond = torch.cat((self.encode_state(x, xref), z), dim=-1)
        return uref + self.feedback(cond, self.normalize_delta(self.error(x, xref)))

    def forward(
        self,
        x: torch.Tensor,
        xref_seq: torch.Tensor,
        uref: torch.Tensor,
        deterministic: bool = False,
    ):
        """Encode the reference window, then act on the tracking error.

        Only ``uref[t]`` is used, as the feedforward term of ``u = uref + W2 tanh(W1 e)``; the
        rest of the ``uref`` window never enters the encoder.
        """
        z = self.encode_reference(x, xref_seq)
        return self.sample(self.mean(x, xref_seq[:, 0], uref, z), deterministic)


class RecurrentCritic(_StateAware):
    """Value function conditioned on the state and the whole reference *path* ahead.

    It mirrors :class:`RecurrentActor`: the same GRU encoding of the same preview window, so actor
    and critic see the reference through the same lens instead of the critic flattening a
    thousand-dimensional observation into one dense layer.

    Its inputs are exactly the state and the reference path -- ``x`` and ``xref``, nothing else.
    Not ``uref``: the reference controls are what the *actor* needs, because its job is to correct
    them, whereas the value of sitting at ``x`` relative to a path is a property of the path's
    geometry.  And not the time index: the value of a (state, path-ahead) situation should not
    depend on the clock.

    Its value head sees ``(x', z)`` and nothing else -- the same conditioning the actor's weight
    generators use, down to encoding the state's positions relative to ``xref[t]``, so the two read
    the situation identically.  Angles arriving here are already wrapped by
    :meth:`~policy.base.BasePolicy.trim_state`.

    Args:
        x_dim: state dimension.
        ref_dim: width of the GRU context.
        hidden_dim: hidden widths of the value head.
        angle_mask: which states are circular; see :mod:`envs.angles`.
    """

    def __init__(
        self,
        x_dim: int,
        ref_dim: int = 64,
        hidden_dim: list = (256, 256),
        activation: nn.Module = nn.Tanh(),
        angle_mask=None,
        x_min=None,
        x_max=None,
        pos_dimension: int = 0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.ref_dim = ref_dim

        self.ref_encoder = nn.GRU(x_dim, ref_dim, batch_first=True)
        # [ x | path context ]
        self.model = MLP(
            x_dim + ref_dim,
            list(hidden_dim),
            1,
            activation=activation,
            initialization="critic",
        )
        self.register_state_geometry(x_dim, angle_mask, x_min, x_max, pos_dimension)

    def encode_reference(self, x: torch.Tensor, xref_seq: torch.Tensor) -> torch.Tensor:
        """Summarise the reference path ``(n, H, x_dim)`` into a context vector ``(n, ref_dim)``.

        Positions relative to ``x`` and the window reversed, for the same reasons as
        :meth:`RecurrentActor.encode_reference`.
        """
        _, h_last = self.ref_encoder(self.encode_window(x, xref_seq).flip(1))
        return h_last.squeeze(0)

    def forward(self, x: torch.Tensor, xref_seq: torch.Tensor) -> torch.Tensor:
        """Value of being at ``x`` given the reference path ``xref_seq`` ahead.

        As in the actor, the state's positions are encoded relative to the current reference point,
        so the value of a tracking situation does not depend on where in the arena it occurs.
        """
        context = self.encode_reference(x, xref_seq)
        return self.model(torch.cat((self.encode_state(x, xref_seq[:, 0]), context), dim=-1))
