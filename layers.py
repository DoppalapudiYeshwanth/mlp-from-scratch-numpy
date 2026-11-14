"""Trainable layers implemented with NumPy only (batch-first conventions)."""

from __future__ import annotations
import numpy as np


class Dense:
    """Fully connected layer computing ``y = x @ W + b``.
    Weights are initialized with a variance chosen to keep activation
    magnitudes roughly constant from layer to layer:
    * ``"he"``: ``std = sqrt(2 / fan_in)``, suited to ReLU networks.
    * ``"xavier"``: ``std = sqrt(1 / fan_in)``, suited to sigmoid/softmax.
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he") -> None:
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"Unknown init {init!r}; expected 'he' or 'xavier'.")

        self.params: dict[str, np.ndarray] = {
            "W": np.random.randn(in_features, out_features) * scale,
            # Bias is (1, out) so it broadcasts across the batch dimension.
            "b": np.zeros((1, out_features)),
        }
        self.grads: dict[str, np.ndarray] = {
            name: np.zeros_like(value) for name, value in self.params.items()
        }
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Computes the affine transform.
        Args:
            x: Input of shape ``(N, in_features)``.
            training: If True, cache ``x`` for the backward pass.\
        Returns:
            Output of shape ``(N, out_features)``.
        """
        
        x = np.asarray(x)
        if training:
            self.cache = {"x": x}
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Computes parameter gradients and the gradient w.r.t. the input.
        Args:
            grad_output: Upstream gradient of shape ``(N, out_features)``
        Returns:
            Gradient with respect to ``x`` of shape ``(N, in_features)``.
        Raises:
            RuntimeError: If called before a training-mode ``forward``.
        """

        if "x" not in self.cache:
            raise RuntimeError("forward(training=True) must be called before backward().")
        x = self.cache["x"]
        # No extra 1/N here: the loss already averages over the batch.
        self.grads["W"] = x.T @ grad_output
        # The bias was broadcast over N rows, so its gradient sums over them.
        self.grads["b"] = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.params["W"].T