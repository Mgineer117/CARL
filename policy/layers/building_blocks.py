# =================================================== #
# Author: Minjae Cho                                  #
# Email: minjae5@illinois.edu                         #
# Affiliation: U of Illinois @ Urbana-Champaign       #
# =================================================== #
"""The one network primitive everything else is built from."""

from typing import Optional, Sequence

import torch
import torch.nn as nn

#: Xavier gain per activation type.  Keyed by class, because ``nn.Module`` does not define
#: ``__eq__``, so comparing activation *instances* would always be False and silently leave the
#: gain at 1.0 for every network.
_GAINS = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.Tanh: "tanh",
    nn.Sigmoid: "sigmoid",
    nn.SELU: "selu",
}


class MLP(nn.Module):
    """Multi-layer perceptron with activation-aware initialisation.

    Args:
        input_dim: input width.
        hidden_dims: hidden widths.
        output_dim: output width; if ``None`` the last hidden layer *is* the output (used when a
            caller attaches its own heads).
        activation: applied after every hidden layer.
        initialization: ``"default"`` (Xavier, gain from the activation), or ``"actor"`` /
            ``"critic"`` (orthogonal, with a small output gain for the actor so the initial policy
            is close to zero).
        dropout_rate: dropout after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        initialization: str = "default",
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ):
        super().__init__()
        gain_name = _GAINS.get(type(activation))
        gain = nn.init.calculate_gain(gain_name) if gain_name else 1.0

        orthogonal = initialization in ("actor", "critic")
        # Orthogonal schemes use sqrt(2) on hidden layers regardless of activation; the output
        # layer of an actor gets a small gain so the initial policy is near zero.
        hidden_gain = 2**0.5 if orthogonal else gain
        out_gain = {"actor": 0.01, "critic": 1.0}.get(initialization, gain)

        widths = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(widths[:-1], widths[1:]):
            layers.append(_init(nn.Linear(in_dim, out_dim), orthogonal, hidden_gain, bias=0.1))
            if activation is not None:
                layers.append(activation)
            if dropout_rate is not None:
                layers.append(nn.Dropout(p=dropout_rate))

        self.output_dim = widths[-1]
        if output_dim is not None:
            layers.append(_init(nn.Linear(widths[-1], output_dim), orthogonal, out_gain))
            self.output_dim = output_dim

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network."""
        return self.model(x)


def _init(layer: nn.Linear, orthogonal: bool, gain: float, bias: float = 0.0) -> nn.Linear:
    if orthogonal:
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    else:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, bias)
    return layer
