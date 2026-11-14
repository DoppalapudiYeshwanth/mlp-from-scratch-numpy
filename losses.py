"""Loss functions implemented with NumPy only (batch-first conventions)."""

from __future__ import annotations
import numpy as np


class MSELoss:
    """Mean squared error averaged over all ``N * C`` elements."""

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the mean squared error.
        Args:
            y_pred: Predictions of shape ``(N, C)``.
            y_true: Targets of shape ``(N, C)``, e.g. one-hot labels.
        Returns:
            Scalar float loss ``mean((y_pred - y_true) ** 2)``.
        Raises:
            ValueError: If ``y_pred`` and ``y_true`` shapes differ.
        """

        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}."
            )
        self.cache = {"y_pred": y_pred, "y_true": y_true}
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to ``y_pred``.
        Returns:
            Gradient of shape ``(N, C)``, equal to
            ``2 * (y_pred - y_true) / (N * C)``.
        Raises:
            RuntimeError: If called before ``forward``.
        """

        if "y_pred" not in self.cache:
            raise RuntimeError("forward() must be called before backward().")
        y_pred = self.cache["y_pred"]
        y_true = self.cache["y_true"]
        n, c = y_pred.shape
        # The mean runs over N * C elements, so the gradient is scaled by both.
        return (2.0 / (n * c)) * (y_pred - y_true)


class CrossEntropyLoss:
    """Categorical cross-entropy on probabilities, averaged over the batch.
    Expects probabilities (e.g. from ``Softmax``). When training with logits,
    prefer ``SoftmaxCrossEntropy``, which is more stable and cheaper.
    """

    _EPS = 1e-12

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes mean cross-entropy loss.
        Args:
            y_pred: Predicted probabilities of shape ``(N, C)``.
            y_true: One-hot targets of shape ``(N, C)``.
        Returns:
            Scalar float loss ``-sum(y_true * log(clip(y_pred))) / N``.
        Raises:
            ValueError: If ``y_pred`` and ``y_true`` shapes differ.
        """

        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}."
            )
        n = y_pred.shape[0]
        # Clip so log(0) can never produce -inf or NaN.
        y_pred_clipped = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        self.cache = {"y_pred_clipped": y_pred_clipped, "y_true": y_true}
        return float(-np.sum(y_true * np.log(y_pred_clipped)) / n)

    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to ``y_pred``.
        Returns:
            Gradient of shape ``(N, C)``, equal to
            ``-y_true / (clip(y_pred) * N)``.
        Raises:
            RuntimeError: If called before ``forward``.
        """

        if "y_pred_clipped" not in self.cache:
            raise RuntimeError("forward() must be called before backward().")
        y_pred_clipped = self.cache["y_pred_clipped"]
        y_true = self.cache["y_true"]
        n = y_pred_clipped.shape[0]
        return -y_true / (y_pred_clipped * n)


class SoftmaxCrossEntropy:
    """Fused softmax + cross-entropy loss operating directly on logits.
    Fusing the two steps gives the simple gradient ``(probs - y_true) / N``
    and avoids the unstable division by tiny probabilities that a separate
    softmax followed by cross-entropy would need.
    """

    _EPS = 1e-12

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Computes softmax probabilities and the mean cross-entropy loss.
        Args:
            logits: Raw scores of shape ``(N, C)``.
            y_true: One-hot targets of shape ``(N, C)``.
        Returns:
            Scalar float mean cross-entropy loss.
        Raises:
            ValueError: If ``logits`` and ``y_true`` shapes differ.
        """

        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true)
        if logits.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs y_true {y_true.shape}."
            )
        n = logits.shape[0]
        # Max-subtraction keeps exp() from overflowing without changing the result.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.cache = {"probs": probs, "y_true": y_true}
        # Clip only for the log; the gradient uses the unclipped probabilities.
        probs_clipped = np.clip(probs, self._EPS, 1.0 - self._EPS)
        return float(-np.sum(y_true * np.log(probs_clipped)) / n)

    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to ``logits``.
        Returns:
            Gradient of shape ``(N, C)``, equal to ``(probs - y_true) / N``.
        Raises:
            RuntimeError: If called before ``forward``.
        """
        
        if "probs" not in self.cache:
            raise RuntimeError("forward() must be called before backward().")
        probs = self.cache["probs"]
        y_true = self.cache["y_true"]
        n = probs.shape[0]
        return (probs - y_true) / n