import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from model import Sequential
from layers import Dense
from activations import ReLU
from losses import SoftmaxCrossEntropy
from optimizers import Adam


def load_data():
    try:
        X, y = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', return_X_y=True)
    except Exception:
        X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32)
    X /= X.max()
    y = y.astype(int)
    Y = np.eye(y.max() + 1, dtype=np.float32)[y]
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n = int(0.8 * len(X))
    tr, va = idx[:n], idx[n:]
    return X[tr], Y[tr], X[va], Y[va]


def get_batches(X, Y, batch_size=64):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        b = idx[i:i + batch_size]
        yield X[b], Y[b]


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_data()
    in_dim, num_classes = X_train.shape[1], Y_train.shape[1]
    model = Sequential([
        Dense(in_dim, 128, 'he'), ReLU(),
        Dense(128, 64, 'he'), ReLU(),
        Dense(64, num_classes, 'xavier'),
    ])
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(lr=0.001)
    epochs = 10
    for e in range(1, epochs + 1):
        total, count = 0.0, 0
        for xb, yb in get_batches(X_train, Y_train, 64):
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            opt.step(model)
            total += loss * len(xb)
            count += len(xb)
        train_acc = (model.forward(X_train).argmax(1) == Y_train.argmax(1)).mean()
        val_acc = (model.forward(X_val).argmax(1) == Y_val.argmax(1)).mean()
        print(f"Epoch {e}/10 | Loss: {total / count:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    dense = [l for l in model.layers if isinstance(l, Dense)]
    np.savez('weights.npz', **{k: v for i, l in enumerate(dense) for k, v in ((f'W{i}', l.params['W']), (f'b{i}', l.params['b']))})