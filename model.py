"""A minimal sequential container for stacking layers."""

from __future__ import annotations
from typing import Sequence
import numpy as np


class Sequential:
    """Chains layers so each one's output feeds the next.
    Every layer must implement ``forward(x, training)`` and ``backward(grad)``.
    """

    def __init__(self, layers: Sequence) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Runs ``x`` through every layer in order.
        Args:
            x: Input batch of shape ``(N, D)``.
            training: If False, layers skip caching intermediates, which
                saves memory at inference time but disables ``backward``.
        Returns:
            Output of the final layer.
        """

        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagates through the layers in reverse order.
        Args:
            grad_output: Gradient of the loss w.r.t. the network output.
        Returns:
            Gradient of the loss w.r.t. the network input.
        """
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def __iter__(self):
        return iter(self.layers)