"""Gradient-based optimizers implemented with NumPy only."""

from __future__ import annotations
from typing import Iterator
import numpy as np


def _iter_params(model) -> Iterator[tuple[tuple[int, str], np.ndarray, np.ndarray]]:
    """Yields ``(key, param, grad)`` for every trainable array in the model.
    Layers without ``params`` (e.g. activations) are skipped. The key is
    stable across steps, so optimizers can store per-parameter state under it.
    """
    for i, layer in enumerate(model.layers):
        params = getattr(layer, "params", None)
        if not params:
            continue
        for name, param in params.items():
            yield (i, name), param, layer.grads[name]


class SGD:
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self, model) -> None:
        """Updates all parameters in place using the stored gradients."""
        for key, param, grad in _iter_params(model):
            v = self._velocity.get(key, np.zeros_like(param))
            # The velocity is a decaying running sum of past gradients.
            v = self.momentum * v - self.lr * grad
            self._velocity[key] = v
            param += v


class Adam:
    """Adam optimizer (Kingma & Ba, 2015)."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[tuple[int, str], np.ndarray] = {}
        self._v: dict[tuple[int, str], np.ndarray] = {}

    def step(self, model) -> None:
        """Updates all parameters in place using the stored gradients."""
        self.t += 1
        for key, param, grad in _iter_params(model):
            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            # Exponential moving averages of the gradient and squared gradient.
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * grad**2
            self._m[key], self._v[key] = m, v

            # m and v start at zero, so early estimates are biased toward zero.
            # Dividing by (1 - beta^t) corrects this; the factor fades as t grows.
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            # Per-parameter step size: large-gradient directions take smaller steps.
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)