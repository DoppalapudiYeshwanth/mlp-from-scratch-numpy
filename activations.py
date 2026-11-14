"""Activation layers implemented with NumPy only (batch-first conventions)."""

from __future__ import annotations
import numpy as np


class ReLU:
    """Rectified Linear Unit activation: ``f(x) = max(0, x)``."""

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Applies ReLU element-wise.
        Args:
            x: Input array of shape ``(N, D)``.
            training: If True, cache the positivity mask for backward.
        Returns:
            Array of shape ``(N, D)`` with negative entries set to zero.
        """

        x = np.asarray(x)
        if training:
            # A boolean mask is all backward needs, and it is cheaper than keeping x.
            self.cache = {"mask": x > 0}
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagates through ReLU.
        Args:
            grad_output: Upstream gradient of shape ``(N, D)``.
        Returns:
            Gradient with respect to ``x`` of shape ``(N, D)``.
        Raises:
            RuntimeError: If called before a training-mode ``forward``.
        """

        if "mask" not in self.cache:
            raise RuntimeError("forward(training=True) must be called before backward().")
        return np.asarray(grad_output) * self.cache["mask"]


class Sigmoid:
    """Numerically stable logistic sigmoid activation."""

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Applies the sigmoid element-wise.
        Args:
            x: Input array of shape ``(N, D)``.
            training: If True, cache the output for backward.
        Returns:
            Array of shape ``(N, D)`` with values in ``(0, 1)``.
        """
        # exp(-x) overflows for large negative x; clipping loses nothing
        # because sigmoid is already saturated at +/-500.

        x_clipped = np.clip(np.asarray(x, dtype=np.float64), -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-x_clipped))
        if training:
            self.cache = {"s": s}
        return s

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagates through the sigmoid.
        Args:
            grad_output: Upstream gradient of shape ``(N, D)``.
        Returns:
            Gradient with respect to ``x`` of shape ``(N, D)``.
        Raises:
            RuntimeError: If called before a training-mode ``forward``.
        """

        if "s" not in self.cache:
            raise RuntimeError("forward(training=True) must be called before backward().")
        s = self.cache["s"]
        # Derivative is expressed through the output: s * (1 - s).
        return np.asarray(grad_output) * s * (1.0 - s)


class Softmax:
    """Row-wise softmax activation with max-subtraction stabilization."""

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Computes softmax probabilities along ``axis=1``.
        Args:
            x: Logits of shape ``(N, C)``.
            training: If True, cache the probabilities for backward.
        Returns:
            Probabilities of shape ``(N, C)``; each row sums to 1.
        """
        x = np.asarray(x, dtype=np.float64)
        # Subtracting the row max leaves softmax unchanged (it cancels in the
        # ratio) but keeps exp() from overflowing on large logits.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        s = exp / np.sum(exp, axis=1, keepdims=True)
        if training:
            self.cache = {"s": s}
        return s

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Computes the per-sample Jacobian-vector product of softmax.
        For each sample, ``J = diag(s) - s s^T`` (symmetric), so
        ``J @ g = s * (g - sum(g * s))``, evaluated for all samples at once.
        Args:
            grad_output: Upstream gradient of shape ``(N, C)``.
        Returns:
            Gradient with respect to ``x`` of shape ``(N, C)``.
        Raises:
            RuntimeError: If called before a training-mode ``forward``.
        """
        
        if "s" not in self.cache:
            raise RuntimeError("forward(training=True) must be called before backward().")
        s = self.cache["s"]
        g = np.asarray(grad_output)
        # (N, 1) dot product broadcasts against (N, C), which avoids
        # building a full (C, C) Jacobian per sample.
        dot = np.sum(g * s, axis=1, keepdims=True)
        return s * (g - dot)