"""Numerical gradient check for the Dense layers of a small network."""

import numpy as np

from activations import ReLU
from layers import Dense
from losses import SoftmaxCrossEntropy
from model import Sequential


def check_gradients(eps: float = 1e-5, tol: float = 1e-6) -> None:
    """Compares analytic gradients against centered finite differences.

    Raises:
        AssertionError: If any parameter's relative error exceeds ``tol``.
    """
    np.random.seed(42)
    net = Sequential([Dense(4, 5, init="xavier"), ReLU(), Dense(5, 3, init="xavier")])
    loss_fn = SoftmaxCrossEntropy()
    X = np.random.randn(6, 4)
    Y = np.eye(3)[np.random.randint(0, 3, 6)]

    # One real forward/backward pass to get the analytic gradients.
    loss_fn.forward(net.forward(X, training=True), Y)
    net.backward(loss_fn.backward())

    def loss() -> float:
        return loss_fn.forward(net.forward(X, training=True), Y)

    max_err = 0.0
    for layer in net.layers:
        if not isinstance(layer, Dense):
            continue
        for name in ("W", "b"):
            p = layer.params[name]
            # Copy first: the loss() calls below don't touch grads, but this
            # guards against any later change that would.
            ana_all = layer.grads[name].copy()
            for idx in np.ndindex(p.shape):
                old = p[idx]
                # Centered difference: O(eps^2) error vs O(eps) for one-sided.
                p[idx] = old + eps
                lp = loss()
                p[idx] = old - eps
                lm = loss()
                p[idx] = old  # restore before moving to the next entry

                num = (lp - lm) / (2 * eps)
                ana = ana_all[idx]
                # Relative error is robust to gradients of very different scales.
                err = abs(ana - num) / (abs(ana) + abs(num) + 1e-12)
                max_err = max(max_err, err)
                assert err < tol, (name, idx, ana, num, err)

    print(f"Gradient check PASSED! Max relative error: {max_err:.3e}")


if __name__ == "__main__":
    check_gradients()