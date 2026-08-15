# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""Angle handling, shared by the environments and the controllers.

Some states live on a circle rather than on a line.  Two things then stop being true, and both are
silent failures rather than errors:

* **A circular state must wrap, not clip.**  Clipping a heading at :math:`\\pm\\pi` freezes that
  degree of freedom: the vehicle can no longer turn through the branch cut at all.
* **Its tracking error is not a subtraction.**  Headings of ``+3.10`` and ``-3.10`` rad are
  0.08 rad apart, but subtract to 6.20 -- so the reward, the metric and the feedback law all see a
  huge error where there is almost none.

Which states are circular is a property of the *system*, so each environment declares them
(``BaseEnv.ANGLE_DIMS``) and exposes an ``angle_mask``; the controllers receive that same mask, so
both sides of the observation contract agree on what an angle is.

Only the car's heading is circular in this benchmark.  The other systems' angles are bounded well
inside a single turn (``|phi| <= pi/3``), which is a physical limit on the operating region, not a
wrap -- so for them the mask is empty and clipping is the intended behaviour.
"""

import numpy as np
import torch

TWO_PI = 2.0 * np.pi


def wrap_to_pi(value):
    """Fold an angle into ``[-pi, pi)``.  Accepts numpy arrays or torch tensors.

    Differentiable for tensors: the derivative is 1 everywhere except at the cut itself, which is a
    measure-zero set, so a gain ``du/dx`` taken through this is unchanged.
    """
    if isinstance(value, torch.Tensor):
        return torch.remainder(value + np.pi, TWO_PI) - np.pi
    return np.remainder(value + np.pi, TWO_PI) - np.pi


def wrap_angles(state, angle_mask):
    """Wrap the circular components of a state vector, leaving the rest untouched.

    ``state`` may be a numpy array or a torch tensor, and ``angle_mask`` either -- the environment
    holds a numpy mask while a controller holds it as a registered torch buffer, and both call
    this.  A mask with no circular states returns ``state`` unchanged.
    """
    if angle_mask is None:
        return state

    if isinstance(state, torch.Tensor):
        mask = torch.as_tensor(angle_mask, dtype=torch.bool, device=state.device)
        if not bool(mask.any()):
            return state
        return torch.where(mask, wrap_to_pi(state), state)

    mask = np.asarray(
        angle_mask.detach().cpu() if isinstance(angle_mask, torch.Tensor) else angle_mask,
        dtype=bool,
    )
    if not mask.any():
        return state

    wrapped = np.array(state, copy=True)
    wrapped[..., mask] = wrap_to_pi(wrapped[..., mask])
    return wrapped


def tracking_error(x, xref, angle_mask):
    """``x - xref``, with circular components reduced to their shortest angular distance.

    This is *the* definition of tracking error in this codebase -- the environment's reward, the
    evaluator's mAUC, the contraction reward and every feedback law all go through it, so they
    cannot disagree about what "far from the reference" means.
    """
    return wrap_angles(x - xref, angle_mask)
