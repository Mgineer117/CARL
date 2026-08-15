# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Batched autograd primitives used by the contraction losses.

Every tensor here is a *batched field*: row ``i`` is an independent sample, so the derivative of
row ``i`` never depends on row ``j``.  Two derivatives are needed (see :mod:`policy.contraction`):

``jacobian(y, x)``
    the full Jacobian ``dy/dx``, shape ``(n, y_dim, x_dim)``.

``directional_derivative(Y, x, [v, ...])``
    for a matrix field ``Y(x)`` and tangent fields ``v(x)``, the derivative of ``Y`` along ``v``,
    ``(partial_v Y)_ij = sum_k dY_ij/dx_k * v_k``, shape ``(n, *Y.shape[1:])`` per tangent.

Why ``directional_derivative`` is written the way it is
-------------------------------------------------------
The obvious implementation materialises the whole derivative tensor ``dY_ij/dx_k`` and contracts
it with ``v``.  For the metric ``Y = W(x)`` that is ``x_dim ** 2`` reverse passes through the
metric network, each retaining a graph for the second-order backward -- and the contraction loss
needs several such calls per step.  At ``x_dim = 10`` that is 100 graph-retaining backward passes
per call, which is what used to dominate both runtime and peak memory.

A directional derivative is just a Jacobian-vector product, and a JVP can be extracted from two
reverse passes ("reverse-over-reverse"):

    g(s) = J(x)^T s              is linear in the cotangent s   -- one reverse pass
    d/ds < g(s), v > = J(x) v    is the JVP we want             -- one more reverse pass

The first pass does not depend on ``v``, so it is shared across tangents and the total cost is
``1 + len(tangents)`` reverse passes *independent of the size of Y*.  The result is still an
ordinary autograd node, so the metric parameters keep receiving gradients through it.
"""

import torch
from torch.autograd import grad


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Full Jacobian ``dy/dx`` of a batched vector field.

    Args:
        y: ``(n, y_dim)`` output field, built from ``x``.
        x: ``(n, x_dim)`` input, must have ``requires_grad=True``.
        create_graph: keep the graph so the Jacobian itself stays differentiable.

    Returns:
        ``(n, y_dim, x_dim)`` tensor with ``J[:, i, k] = dy_i/dx_k``.
    """
    # Guard against components of `y` that happen to be constant in `x` (e.g. the zero rows of
    # f(x) for the quadrotor): autograd would otherwise complain the input is unused. The added
    # term is exactly zero and depends only on the *own* row of x, so no samples get coupled.
    y = y + 0.0 * x.sum(dim=-1, keepdim=True)

    rows = [
        grad(y[:, i].sum(), x, create_graph=create_graph, retain_graph=True)[0]
        for i in range(y.shape[-1])
    ]
    return torch.stack(rows, dim=1)


def directional_derivative(
    Y: torch.Tensor, x: torch.Tensor, tangents: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Derivatives of the batched field ``Y(x)`` along each tangent field in ``tangents``.

    ``partial_v Y = sum_k (dY/dx_k) v_k``, one output per tangent, each shaped like ``Y``.

    Costs ``1 + len(tangents)`` reverse passes rather than ``Y[0].numel()`` of them; see the
    module docstring.

    Args:
        Y: ``(n, ...)`` field built from ``x`` (typically the metric ``(n, x_dim, x_dim)``).
        x: ``(n, x_dim)`` input with ``requires_grad=True``.
        tangents: tangent fields, each ``(n, x_dim)``.
    """
    # `s` is a dummy cotangent. `JT_s` is linear in it, so its value is irrelevant and zeros
    # keep the intermediate magnitudes small.
    s = torch.zeros_like(Y).requires_grad_()
    (JT_s,) = grad(Y, x, grad_outputs=s, create_graph=True)
    return [grad(JT_s, s, grad_outputs=v, create_graph=True)[0] for v in tangents]


def symmetrise(A: torch.Tensor) -> torch.Tensor:
    """``(A + A^T) / 2`` for a batch of square matrices."""
    return 0.5 * (A + A.transpose(-1, -2))
