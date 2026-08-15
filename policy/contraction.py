# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""The contraction certificate: the CCM conditions, their diagnostics, and the training mixin.

Three pieces, in dependency order:

:class:`ContractionLoss`
    the CCM conditions as one differentiable scalar;
:class:`ContractionDiagnostics`
    the eigenvalue history that says whether those conditions actually *hold*;
:class:`MetricLearner`
    the mixin that owns both, plus the metric optimiser, the offline batch sampler and the warmup
    phase -- everything CARL needs to learn a metric.

The contraction math therefore has exactly one implementation, independent of how the controller
is trained.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from policy.autograd import directional_derivative, jacobian, symmetrise


class ContractionLoss:
    r"""Control-contraction-metric (CCM) conditions turned into a scalar loss.

    For a control-affine system

    .. code-block:: text

        xdot = f(x) + B(x) u

    a feedback law :math:`u = \pi(x, \cdot)` and the *dual* metric :math:`W(x) = M(x)^{-1}`, the
    closed loop contracts at rate ``lbd`` in the metric ``M`` if, pointwise in ``x``,

    .. code-block:: text

        (Cu)  Mdot + 2 sym( M (A + B K) ) + 2 lbd M                          <  0
        (C1)  Bbot^T ( -partial_f W + 2 sym( (df/dx) W ) + 2 lbd W ) Bbot    <  0
        (C2)  Bbot^T ( partial_{b_j} W - 2 sym( (dB_j/dx) W ) ) Bbot         =  0   for each j

    where

    .. code-block:: text

        A     = df/dx + sum_j u_j dB_j/dx      state Jacobian of the dynamics at frozen u
        K     = d(pi)/dx                       feedback gain
        Mdot  = partial_{f + B u} M            M differentiated along the closed-loop flow
        Bbot  with B^T Bbot = 0                annihilator of the control directions
        partial_v Y = sum_k (dY/dx_k) v_k      derivative of a field Y along a vector field v

    ``(Cu)`` is the closed-loop contraction condition proper.  ``(C1)`` and ``(C2)`` are what make
    the metric a *control* contraction metric: they constrain ``W`` only in the directions the
    control cannot reach, and so can be enforced without committing to a particular controller --
    which is what lets the metric be learned alongside the policy, and what the warmup phase
    (:meth:`MetricLearner.warmup_metric`) optimises on its own.

    Every occurrence of the metric carries gradient, including the ``2 lbd M`` and ``2 lbd W``
    terms: the loss is then exactly the gradient of the stated constraint, with no surrogate.

    Each matrix inequality becomes a scalar via :meth:`_pd_penalty`; ``(C2)`` is an equality and is
    penalised in squared Frobenius norm.  Two further terms keep the metric well conditioned:
    ``W >= w_lb I`` holds by construction (the lower bound is added here), and ``W <= w_ub I`` is
    requested through the ``overshoot`` penalty.

    Cost per call
    -------------
    * ``1 + action_dim`` Jacobians of the dynamics -- for ``df/dx`` and each ``dB_j/dx``;
    * ``action_dim`` Jacobian rows of the policy -- for ``K``;
    * ``2 + action_dim`` reverse passes through the metric network *in total*, for every
      directional derivative of ``W``.  Materialising ``dW/dx`` instead would cost
      ``(2 + action_dim) * x_dim ** 2`` graph-retaining passes, which is where essentially all of
      the runtime and peak memory of this loss used to go.  See :mod:`policy.autograd`.

    Under ``warmup=True`` the second item disappears entirely, along with the batched inverse
    ``M = W^-1``: ``(Cu)`` is the only condition that mentions the controller, so nothing that
    exists to build it is computed.

    Args:
        metric_net: metric generator, ``x -> (W_raw, info)`` with ``W_raw`` symmetric PSD.
        get_f_and_B: dynamics oracle, ``x -> (f, B, Bbot)``.  Either the environment's analytic
            functions or a learned :class:`~policy.layers.dynamics.DynamicLearner`.
        x_dim: state dimension.
        action_dim: control dimension.
        lbd: desired contraction rate.
        eps: margin by which the strict inequalities must hold.
        w_lb: lower bound added to the metric, ``W = W_raw + w_lb I``.
        w_ub: requested upper bound on the metric.
        num_probes: random directions used to test each matrix inequality; see
            :meth:`_pd_penalty`.
        metric_entropy_scaler: weight of an entropy bonus on the (stochastic) metric generator.
            The conditions are then enforced over a *neighbourhood* of metrics rather than at a
            single point estimate, because nothing else in this loss stops the generator's
            variance collapsing.  Zero disables it, and it has no effect on a deterministic
            generator or when the metric is evaluated at its mean.
        dtype: dtype the dynamics oracle's outputs are cast to.
    """

    def __init__(
        self,
        metric_net: nn.Module,
        get_f_and_B: Callable,
        x_dim: int,
        action_dim: int,
        lbd: float = 0.5,
        eps: float = 0.1,
        w_lb: float = 1e-1,
        w_ub: float = 10.0,
        num_probes: int = 1024,
        metric_entropy_scaler: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.metric_net = metric_net
        self.get_f_and_B = get_f_and_B
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.lbd = lbd
        self.eps = eps
        self.w_lb = w_lb
        self.w_ub = w_ub
        self.num_probes = num_probes
        self.metric_entropy_scaler = metric_entropy_scaler
        self._dtype = dtype

    # ------------------------------------------------------------------ metric

    def metric(self, x: torch.Tensor, deterministic: bool = False, inverse: bool = True):
        """Return ``(W, M)``: the dual metric with its lower bound folded in, and its inverse.

        ``deterministic`` selects the mean of a stochastic metric generator.  CARL trains the
        generator on samples but *rewards* with the mean, so this flag is what separates the two
        uses; see :class:`~policy.layers.metrics.MetricNetwork`.

        ``inverse=False`` returns ``M = None`` and skips the batched inverse.  Only ``(Cu)`` needs
        ``M``, so the warmup phase -- which drops ``(Cu)`` -- would otherwise pay a cubic
        factorisation per sample for a matrix it never reads.

        Returns ``(W, M, info)``; ``info`` carries the generator's distribution, log-probability
        and entropy, the last of which the loss rewards.
        """
        W_raw, info = self.metric_net(x, deterministic=deterministic)
        W = W_raw + self.w_lb * _eye_like(W_raw, self.x_dim)
        return W, (torch.linalg.inv(W) if inverse else None), info

    # ------------------------------------------------------------------ loss

    def __call__(
        self,
        x: torch.Tensor,
        policy_mean: Callable[[torch.Tensor], torch.Tensor],
        warmup: bool = False,
        deterministic: bool = False,
        detach_policy: bool = True,
    ):
        """Evaluate the contraction loss on a batch of states.

        Args:
            x: ``(n, x_dim)`` states; must have ``requires_grad=True``.
            policy_mean: ``x -> u`` giving the *deterministic* control whose closed loop is being
                certified.  Certifying a sampled action instead would put exploration noise into
                the certificate.
            warmup: keep only the controller-independent conditions ``(C1)``, ``(C2)`` and the
                metric bounds, dropping the closed-loop condition ``(Cu)``.  This is the pretraining
                objective: it makes the metric a valid CCM before it is asked to certify any
                particular policy.
            deterministic: evaluate the metric at its mean rather than sampling it.
            detach_policy: cut the gradient to the controller, so this loss trains the metric only.
                CARL detaches: its controller is trained by RL, and the metric merely certifies
                whatever policy it currently has.  Passing ``False`` instead makes this loss the
                controller's objective as well.

        Returns:
            ``(loss, terms, matrices)`` -- the scalar loss, a dict of its named components, and the
            raw matrices for :class:`ContractionDiagnostics`.
        """
        n_u = self.action_dim
        # (Cu) is the only condition that mentions the controller, so warmup skips everything that
        # exists solely to build it: the policy forward, the gain `K` -- whose Jacobian runs back
        # through every step of the actor's recurrent preview encoder, by far the most expensive
        # term here -- and the batched inverse `M`.  Computing them and then dropping them on the
        # floor cost 3-4x the whole warmup phase.
        certify_policy = not warmup

        W, M, metric_info = self.metric(x, deterministic=deterministic, inverse=certify_policy)

        f, B, Bbot = self._dynamics(x)

        if certify_policy:
            # `u` must carry a gradient w.r.t. x for K to exist at all; detaching happens *after*
            # the Jacobian, so cutting the controller loose never costs us the gain itself.
            u = policy_mean(x)
            K = jacobian(u, x)  # (n, action_dim, x_dim)
            if detach_policy:
                u, K = u.detach(), K.detach()

        # --- derivatives of the dynamics ------------------------------------------------
        DfDx = jacobian(f, x)  # (n, x_dim, x_dim)
        DBDx = [jacobian(B[:, :, j], x) for j in range(n_u)]  # each (n, x_dim, x_dim)

        f, B, Bbot = f.detach(), B.detach(), Bbot.detach()

        # --- derivatives of the metric --------------------------------------------------
        # partial_v W is linear in v, so the closed-loop derivative is a combination of the ones
        # already needed by (C1) and (C2):
        #     partial_{f + B u} W = partial_f W + sum_j u_j partial_{b_j} W
        # which means action_dim + 1 tangents cover every term below.
        DfW, *DbW = directional_derivative(W, x, [f, *(B[:, :, j] for j in range(n_u))])

        # --- (C1), (C2): the conditions restricted to the annihilator of B ---------------
        BbotT = Bbot.transpose(1, 2)
        C1 = BbotT @ (-DfW + 2 * symmetrise(DfDx @ W) + 2 * self.lbd * W) @ Bbot
        C2 = [BbotT @ (DbW[j] - 2 * symmetrise(DBDx[j] @ W)) @ Bbot for j in range(n_u)]

        # The inequalities are strict, so ask for a margin of eps.
        C1 = C1 + self.eps * _eye_like(C1, C1.shape[-1])
        overshoot_matrix = W - self.w_ub * _eye_like(W, self.x_dim)

        c1 = self._pd_penalty(-C1)
        overshoot = self._pd_penalty(-overshoot_matrix)
        c2 = sum(c.flatten(1).pow(2).sum(1).mean() for c in C2)

        # Reward the generator's spread.  Sampling `W` is what enforces the conditions over a
        # neighbourhood of metrics, but nothing else in this loss opposes the variance shrinking to
        # zero -- a narrower distribution simply makes every condition easier to satisfy.
        entropy = metric_info["entropy"].mean()
        loss = c1 + c2 + overshoot - self.metric_entropy_scaler * entropy
        terms = {"c1": c1, "c2": c2, "overshoot": overshoot, "generator_entropy": entropy}
        matrices = {"C1": C1, "C2": c2, "overshoot": overshoot_matrix}

        # --- (Cu) closed-loop contraction -----------------------------------------------
        if certify_policy:
            dot_W = DfW + sum(u[:, j].view(-1, 1, 1) * DbW[j] for j in range(n_u))
            dot_M = -M @ dot_W @ M  # d/dt (W^-1) = -W^-1 Wdot W^-1, exact and one less derivative

            A = DfDx + sum(u[:, j].view(-1, 1, 1) * DBDx[j] for j in range(n_u))
            sym_MABK = symmetrise(M @ (A + B @ K))
            Cu = dot_M + 2 * sym_MABK + 2 * self.lbd * M
            Cu = Cu + self.eps * _eye_like(Cu, Cu.shape[-1])

            cu = self._pd_penalty(-Cu)
            loss = loss + cu
            terms["cu"] = cu
            matrices.update({"Cu": Cu, "dot_M": dot_M, "sym_MABK": sym_MABK})
        else:
            terms["cu"] = torch.zeros((), dtype=W.dtype, device=W.device)

        # `matrices` carries the (Cu) entries only when they were computed; the diagnostics that
        # read them are recorded only outside warmup, which is exactly when that holds.
        return loss, terms, matrices

    # ------------------------------------------------------------------ internals

    def _dynamics(self, x: torch.Tensor):
        """``f(x), B(x), Bbot(x)`` cast to the working dtype and device."""
        f, B, Bbot = self.get_f_and_B(x)
        return tuple(torch.as_tensor(t, dtype=self._dtype, device=x.device) for t in (f, B, Bbot))

    def _pd_penalty(self, A: torch.Tensor) -> torch.Tensor:
        """Push the symmetric batch ``A`` towards positive definiteness.

        Eigendecomposing every sample would be cubic and has an ill-behaved gradient at repeated
        eigenvalues, so instead ``A`` is probed with random unit vectors and ``z^T A z < 0`` is
        penalised.  Two details decide whether that surrogate is any good, and both follow the
        reference C3M implementation:

        **Probe count.**  ``num_probes`` directions are drawn and *shared across the batch*, giving
        ``batch x num_probes`` quadratic forms.  A single direction per sample -- what this used to
        do -- is a very weak detector: for random unit ``z``, ``z^T A z = sum_i lbd_i c_i^2`` with
        ``E[c_i^2] = 1/d``, so a matrix with eigenvalues ``(-1, -1, -1, +0.1)`` has
        ``E[z^T A z] = -0.725`` and one draw almost never registers the violation.  A thousand
        directions per matrix do.

        **Normalisation.**  The mean is taken over the *violating* forms only, not over all of
        them.  Averaging over the whole batch makes the penalty -- and its gradient -- fade as
        violations become rare, which is to say the constraint stops being enforced exactly where
        it is nearly satisfied.  Dividing by the violation count keeps the corrective signal at
        full strength however few samples are left to correct.
        """
        z = torch.randn(self.num_probes, A.shape[-1], dtype=A.dtype, device=A.device)
        z = z / z.norm(dim=-1, keepdim=True)
        zAz = torch.einsum("ki,bij,kj->bk", z, A, z)

        violating = zAz < 0
        if not bool(violating.any()):
            return torch.zeros((), dtype=A.dtype, device=A.device)
        return -zAz[violating].mean()


def _eye_like(ref: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.eye(dim, dtype=ref.dtype, device=ref.device)


# ===================================================================== diagnostics


class ContractionDiagnostics:
    """Tracks the eigenvalue spectra of the contraction matrices and plots their history.

    The conditions are matrix inequalities, so the loss alone does not say whether they *hold*:
    what matters is the sign of the extreme eigenvalues of ``Cu`` and ``C1``.  This records the
    per-batch mean spectrum of each matrix and renders it as a training curve.
    """

    KEYS = ("Cu", "dot_M", "sym_MABK", "C1", "overshoot")
    TITLES = {
        "Cu": "Cu eigenvalues (want < 0)",
        "dot_M": "Mdot eigenvalues",
        "sym_MABK": "sym(M(A+BK)) eigenvalues",
        "C1": "C1 eigenvalues (want < 0)",
        "overshoot": "W - w_ub I eigenvalues (want < 0)",
    }

    def __init__(self, stride: int = 10):
        self.stride = stride
        self.history = {k: [] for k in self.KEYS}
        self.c2_history = []

    def record(self, matrices: dict):
        """Append the mean eigenvalue spectrum of this batch's contraction matrices."""
        with torch.no_grad():
            for key in self.KEYS:
                # eigvalsh on the CPU: it is unimplemented on the MPS backend, and this is a
                # logging path where the transfer cost does not matter.
                eig = torch.linalg.eigvalsh(matrices[key].detach().cpu())
                self.history[key].append(eig.mean(0).numpy())
            self.c2_history.append(float(matrices["C2"]))

    def plot(self):
        """Return a figure of the recorded spectra, or ``None`` if there is not enough data yet."""
        if len(self.c2_history) < self.stride:
            return None

        steps = list(range(0, len(self.c2_history), self.stride))
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))

        for ax, key in zip(axes.flat, self.KEYS):
            eig = np.asarray(self.history[key][:: self.stride])
            lo, hi = eig.min(axis=1), eig.max(axis=1)
            ax.plot(steps, eig.mean(axis=1), label=f"max={hi[-1]:.3g}, min={lo[-1]:.3g}")
            ax.fill_between(steps, lo, hi, alpha=0.2)
            ax.set_title(self.TITLES[key])
            ax.legend()

        ax = axes.flat[len(self.KEYS)]
        c2 = np.asarray(self.c2_history[:: self.stride])
        ax.plot(steps, c2, label=f"C2 loss = {c2[-1]:.3g}")
        ax.set_yscale("log")
        ax.set_title("C2 loss (want -> 0)")
        ax.legend()

        for ax in axes.flat:
            ax.grid(linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.close(fig)
        return fig


# ===================================================================== training mixin


class MetricLearner:
    """Everything needed to *learn* a contraction metric.

    Mixed into a :class:`~policy.base.BasePolicy` subclass, it supplies the metric network, the
    contraction loss, the diagnostics, the offline batch sampler and the warmup phase.  The host
    class decides only how the *controller* is trained, and calls :meth:`setup_contraction` from
    its own ``__init__``.

    ``metric_optimizer`` trains the metric alone, which is what CARL uses for every metric update.
    """

    #: None of this is needed to *act*, so it is not shipped to the rollout workers.  ``data`` in
    #: particular holds the whole offline dataset.
    TRAINING_ONLY_ATTRS = ("contraction", "diagnostics", "data")

    def setup_contraction(
        self,
        metric_net: nn.Module,
        get_f_and_B: Callable,
        data,
        metric_lr: float,
        batch_size: int,
        lbd: float,
        eps: float,
        w_lb: float,
        w_ub: float,
        num_probes: int = 1024,
        metric_entropy_scaler: float = 0.0,
        max_grad_norm: float = 10.0,
    ):
        """Attach the metric machinery.  Call from the host class's ``__init__``."""
        self.metric_net = metric_net
        self.data = data
        self.metric_batch_size = batch_size
        self.metric_max_grad_norm = max_grad_norm

        # A learned dynamics model reaches this loss only through `self.contraction`, which is a
        # plain object rather than an nn.Module -- so it is never registered as a child, and
        # `self.train()` cannot switch it back into training mode and put dropout noise into every
        # contraction condition.  Freezing it here makes that explicit.
        if isinstance(get_f_and_B, nn.Module):
            get_f_and_B.eval()
            get_f_and_B.requires_grad_(False)

        self.contraction = ContractionLoss(
            metric_net=metric_net,
            get_f_and_B=get_f_and_B,
            x_dim=self.x_dim,
            action_dim=self.action_dim,
            lbd=lbd,
            eps=eps,
            w_lb=w_lb,
            w_ub=w_ub,
            num_probes=num_probes,
            metric_entropy_scaler=metric_entropy_scaler,
            dtype=self._dtype,
        )
        self.diagnostics = ContractionDiagnostics()
        self.metric_optimizer = torch.optim.Adam(self.metric_net.parameters(), lr=metric_lr)

    # ------------------------------------------------------------------ batches

    def sample_contraction_batch(self, size: int | None = None):
        """Draw offline ``(x, xref_seq, uref)``; ``x`` carries the gradient the loss needs."""
        batch = self.data.sample(size or self.metric_batch_size)
        return (
            self.to_tensor(batch["x"]).requires_grad_(),
            self.to_tensor(batch["xref"]),
            self.to_tensor(batch["uref"]),
        )

    def policy_mean_fn(self, xref_seq, uref):
        """The control the metric certifies, as a function of ``x`` alone.

        The reference window is captured and held fixed, so the returned closure depends only on
        ``x`` -- which is exactly what ``K = du/dx`` must differentiate.

        For a recurrent actor the preview context is encoded **inside** the closure, not hoisted
        out of it.  The window is encoded relative to ``x``, so ``z`` is a function of ``x``: moving
        the state really does move the encoded reference, and that path is part of the closed-loop
        gain.  Hoisting the encoding out (or detaching it) would certify a gain the closed loop
        does not have -- the conditions would hold for a system nobody is running.  The cost is
        that the Jacobian is backpropagated through ``H`` GRU steps, which is what ``--ref-stride``
        keeps small.
        """
        # uref is a constant of x, so it shifts u without touching K = du/dx -- but it must be
        # present, because the certified control is the one the plant actually receives.
        if hasattr(self.actor, "encode_reference"):
            return lambda x: self.actor.mean(
                x, xref_seq[:, 0], uref, self.actor.encode_reference(x, xref_seq)
            )
        return lambda x: self.actor.mean(x, xref_seq[:, 0], uref)

    # ------------------------------------------------------------------ updates

    def metric_step(self, warmup: bool, record: bool = True):
        """One gradient step on the contraction loss, updating the metric only.

        Returns ``(loss, terms)`` with ``loss`` a float.
        """
        x, xref_seq, uref = self.sample_contraction_batch()

        loss, terms, matrices = self.contraction(
            x, self.policy_mean_fn(xref_seq, uref), warmup=warmup, detach_policy=True
        )
        if record and not warmup:
            self.diagnostics.record(matrices)

        self.metric_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.metric_net.parameters(), self.metric_max_grad_norm)
        self.metric_optimizer.step()
        return loss.item(), terms

    def warmup_metric(self, epochs: int, target_loss: float = 0.01):
        """Pretrain the metric on ``(C1)``, ``(C2)`` and the metric bounds.

        These are the conditions that do not involve the controller, so the metric can be made a
        valid CCM before any policy exists to certify.  Only ``metric_net`` is updated -- the
        controller is untouched even when it is otherwise trained by this same loss.

        ``target_loss`` is an early exit, not a requirement.  With a *stochastic* metric (CARL) the
        sampled ``W`` has a variance floor, so the loss plateaus well above any small threshold and
        warmup simply runs its full budget; that is expected, not a failure to converge.
        """
        if epochs <= 0:
            return

        loss = float("nan")
        with tqdm(range(epochs), desc=f"{self.name} metric warmup", leave=False) as pbar:
            for epoch in pbar:
                loss, _ = self.metric_step(warmup=True)
                pbar.set_postfix(loss=f"{loss:.4f}")
                if loss < target_loss:
                    pbar.write(
                        f"[{self.name}] metric warmup converged at epoch {epoch + 1} "
                        f"(loss {loss:.4f} <= {target_loss})"
                    )
                    return
        print(f"[{self.name}] metric warmup hit the epoch limit at loss {loss:.4f}.")

    # ------------------------------------------------------------------ logging

    def metric_log(self, loss: float, terms: dict, lr: float) -> dict:
        """The metric half of the training log: the contraction total and its named conditions."""
        log = {f"{self.name}/metric/{key}": value.item() for key, value in terms.items()}
        log[f"{self.name}/metric/contraction"] = loss
        log[f"{self.name}/lr/metric"] = lr
        log.update(self.compute_gradient_norm([self.metric_net], ["metric"], prefix=self.name))
        return log

    def metric_plot(self) -> dict:
        """The eigenvalue figure, or an empty dict when there is not enough history yet."""
        fig = self.diagnostics.plot()
        return {f"{self.name}/plot/eigenvalues": fig} if fig is not None else {}
