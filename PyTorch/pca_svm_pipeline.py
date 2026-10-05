import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from poincare_features import time_delay_embedding, get_poincare_intersections, extract_features
from svm_training import fit_scaler, apply_scaler, compute_metrics

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Hyperparameters (full control, vs. sklearn defaults previously)
# ==========================================
LAYER1_EPOCHS = 200      # per-channel classifier (replaces LDA)
LAYER1_LR = 0.05
LAYER2_EPOCHS = 300      # combiner classifier (replaces linear SVC)
LAYER2_LR = 0.05
LAYER2_C = 1.0           # inverse regularization strength, like sklearn's SVC(C=...)

# Layer 2 modes:
#   'svm'   -> supervised "seizure vs non-seizure" linear SVM (original setup)
#   'ocsvm' -> "anomaly vs normal" detector: One-Class SVM trained only on
#              non-seizure epochs; anything outside "normal" is called a seizure
MODES = ['svm', 'ocsvm']

OCSVM_NU = 0.05          # upper bound on the fraction of normal training epochs flagged as anomalies
OCSVM_N_FOURIER = 500    # random Fourier features approximating the RBF kernel
OCSVM_EPOCHS = 500
OCSVM_LR = 0.01


class LinearClassifier(nn.Module):
    """
    A plain linear layer used as a general-purpose binary classifier in this
    demo pipeline - the exact same module architecture is reused for both
    Layer 1 (trained as logistic regression, replacing LDA) and Layer 2
    (trained as a hinge-loss SVM). What makes each one behave differently is
    the loss function used to train it (`train_logistic` vs.
    `train_linear_svm`), not the model itself.
    """

    def __init__(self, in_features):
        """
        Args:
            in_features: number of input features per sample.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor of shape (batch_size,) with one raw decision score
            per sample.
        """
        return self.linear(x).squeeze(-1)


class OneClassSVM(nn.Module):
    """
    PyTorch replacement for sklearn's OneClassSVM(kernel='rbf') - an anomaly
    detector that only ever sees NORMAL data during training. It learns a
    boundary enclosing the normal samples; at inference, anything falling
    outside it is an anomaly (here: a seizure).

    The RBF kernel k(x, x') = exp(-gamma * ||x - x'||^2) is approximated with
    random Fourier features (Rahimi & Recht, 2007):
        phi(x) = sqrt(2 / D) * cos(x @ W + b),  W ~ N(0, 2*gamma), b ~ U(0, 2*pi)
    so that phi(x) . phi(x') ~= k(x, x'). This turns the kernel problem into
    a linear one in phi-space, solvable with gradient descent like the other
    models in this file.

    The input standardization (mean/std of the normal training data) is
    stored inside the module, so callers pass raw Layer 1 outputs.
    """

    def __init__(self, in_features, n_fourier=OCSVM_N_FOURIER):
        """
        Args:
            in_features: number of input features per sample.
            n_fourier: D, the number of random Fourier features (more =
                closer approximation of the exact RBF kernel, but slower).
        """
        super().__init__()
        self.in_features = in_features
        self.n_fourier = n_fourier
        self.w = nn.Parameter(torch.zeros(n_fourier))
        self.rho = nn.Parameter(torch.zeros(()))
        # Fixed (non-trainable) state, set by `train_one_class_svm`
        self.register_buffer('mean', torch.zeros(in_features))
        self.register_buffer('std', torch.ones(in_features))
        self.register_buffer('W', torch.zeros(in_features, n_fourier))
        self.register_buffer('b', torch.zeros(n_fourier))

    def features(self, x):
        """Standardizes x and maps it to the random Fourier feature space."""
        x = apply_scaler(x, self.mean, self.std)
        return math.sqrt(2.0 / self.n_fourier) * torch.cos(x @ self.W + self.b)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor of shape (batch_size,) with the decision score
            w . phi(x) - rho: >= 0 means inlier (normal), < 0 means outlier.
        """
        return self.features(x) @ self.w - self.rho


def train_logistic(model, X, y, epochs, lr):
    """
    Trains a `LinearClassifier` as a logistic-regression model - this is the
    PyTorch replacement for sklearn's LDA in Layer 1. Logistic regression,
    like LDA, learns a single linear decision boundary between two classes;
    unlike LDA (which has a closed-form solution from class means/
    covariances), here it's fit by directly minimizing binary cross-entropy
    with gradient descent, which is what exposes `epochs`/`lr` as tunable.

    Args:
        model: a `LinearClassifier` instance to train in place.
        X: torch.Tensor of shape (n_samples, in_features).
        y: torch.Tensor of shape (n_samples,) with binary labels (0/1).
        epochs: number of full-batch gradient descent steps.
        lr: learning rate for the Adam optimizer.

    Returns:
        The same `model` instance, now trained (in `.eval()` mode).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_float = y.float()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y_float)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_linear_svm(model, X, y, C, epochs, lr):
    """
    Trains a `LinearClassifier` as a linear SVM via the soft-margin hinge
    loss objective (see `svm_training.train_linear_svm` for the same idea
    with fuller commentary) - the PyTorch replacement for Layer 2's
    sklearn SVC(kernel='linear'). Used here to combine Layer 1's 23
    per-channel binary predictions into one final decision.

    Args:
        model: a `LinearClassifier` instance to train in place.
        X: torch.Tensor of shape (n_samples, in_features) - here, Layer 1's
            per-channel predictions (n_samples, n_channels).
        y: torch.Tensor of shape (n_samples,) with binary labels (0/1).
        C: regularization strength - large C fits the training data harder
            (weaker regularization), small C prefers a simpler boundary.
        epochs: number of full-batch gradient descent steps.
        lr: learning rate for the Adam optimizer.

    Returns:
        The same `model` instance, now trained (in `.eval()` mode).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_signed = y.float() * 2 - 1
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        hinge = torch.clamp(1 - y_signed * out, min=0).mean()
        w = model.linear.weight
        loss = 0.5 * (w ** 2).sum() + C * hinge
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_one_class_svm(model, X_normal, nu, epochs, lr, gamma=None):
    """
    Trains a `OneClassSVM` on NORMAL samples only, minimizing the primal
    One-Class SVM objective (Schölkopf et al., 2001):

        0.5 * ||w||^2 - rho + 1/(nu * n) * sum(max(0, rho - w . phi(x_i)))

    i.e. push the hyperplane w . phi(x) = rho as far from the origin as
    possible while keeping (most of) the normal samples on its far side.
    `nu` caps the fraction of training samples allowed on the wrong side, so
    it works as the false-alarm rate you are willing to accept on normal EEG.

    Args:
        model: a `OneClassSVM` instance to train in place.
        X_normal: torch.Tensor of shape (n_samples, in_features) containing
            ONLY non-seizure samples.
        nu: in (0, 1] - fraction of normal training samples treated as
            outliers (like sklearn's OneClassSVM(nu=...)).
        epochs: number of full-batch gradient descent steps.
        lr: learning rate for the Adam optimizer.
        gamma: RBF kernel width. None = sklearn's gamma='scale', i.e.
            1 / (n_features * X.var()) on the standardized data.

    Returns:
        The same `model` instance, now trained (in `.eval()` mode).
    """
    mean, std = fit_scaler(X_normal)
    model.mean.copy_(mean)
    model.std.copy_(std)

    if gamma is None:
        X_std = apply_scaler(X_normal, mean, std)
        var = X_std.var().item()
        gamma = 1.0 / (model.in_features * var) if var > 0 else 1.0
    model.W.normal_(0.0, math.sqrt(2.0 * gamma))
    model.b.uniform_(0.0, 2 * math.pi)

    optimizer = torch.optim.Adam([model.w, model.rho], lr=lr)
    n = X_normal.shape[0]
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        slack = torch.clamp(-model(X_normal), min=0)  # max(0, rho - w . phi(x))
        loss = 0.5 * (model.w ** 2).sum() - model.rho + slack.sum() / (nu * n)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_labels(model, X):
    """
    Predicts binary labels from a trained `LinearClassifier` by thresholding
    its raw decision score at zero (works the same regardless of whether
    the model was trained with `train_logistic` or `train_linear_svm`,
    since both output the same kind of raw score).

    Args:
        model: a trained `LinearClassifier`.
        X: torch.Tensor of shape (n_samples, in_features).

    Returns:
        torch.Tensor of shape (n_samples,), dtype long, with 0/1 predictions.
    """
    with torch.no_grad():
        return (model(X) > 0).long()


def predict_anomalies(model, X):
    """
    Predicts binary labels from a trained `OneClassSVM`, mapped to the same
    convention as `predict_labels`: an outlier (score < 0) is 1 = seizure,
    an inlier is 0 = non-seizure.

    Args:
        model: a trained `OneClassSVM`.
        X: torch.Tensor of shape (n_samples, in_features).

    Returns:
        torch.Tensor of shape (n_samples,), dtype long, with 0/1 predictions.
    """
    with torch.no_grad():
        return (model(X) < 0).long()


# ==========================================
# Main Pipeline & Classification
# ==========================================
def simulate_epoch_features(n_epochs, n_channels, n_samples):
    """
    Runs PSR -> Poincaré -> features for every epoch/channel on SYNTHETIC
    (random noise) signals.

    Returns:
        torch.Tensor of shape (n_epochs, n_channels, 7).
    """
    X = torch.zeros((n_epochs, n_channels, 7), device=DEVICE)
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Replace this with your actual MNE epoch data
            raw_signal = np.random.randn(n_samples)

            embedded = time_delay_embedding(raw_signal, d=5, tau=6)
            intersections = get_poincare_intersections(embedded)
            X[epoch, ch, :] = extract_features(intersections)
    return X


def layer1_outputs(layer1_models, X_all_channels, mode):
    """
    Builds the Layer 2 input from the 23 per-channel Layer 1 classifiers.

    Args:
        layer1_models: list of trained `LinearClassifier`, one per channel.
        X_all_channels: torch.Tensor of shape (n_epochs, n_channels, 7).
        mode: 'svm' -> binary per-channel predictions (as in the paper).
              'ocsvm' -> raw per-channel decision scores. The One-Class SVM
              only sees non-seizure epochs, whose binary votes are almost
              all 0 - a near-degenerate training set. The raw scores keep
              the "how confident" information the anomaly detector needs.

    Returns:
        torch.Tensor of shape (n_epochs, n_channels).
    """
    n_epochs, n_channels, _ = X_all_channels.shape
    out = torch.zeros((n_epochs, n_channels), device=DEVICE)
    with torch.no_grad():
        for ch, model in enumerate(layer1_models):
            X_ch = X_all_channels[:, ch, :]
            out[:, ch] = predict_labels(model, X_ch).float() if mode == 'svm' else model(X_ch)
    return out


def run_pipeline(mode='svm', seed=42):
    """
    Demonstrates the full 2-layer classification architecture end to end on
    SYNTHETIC (random noise) data, to show how the pieces fit together
    without needing real EEG data loaded. Replace the
    `np.random.randn(n_samples)` line in `simulate_epoch_features` with real
    per-channel epoch data to turn this into a real pipeline.

    Architecture:
      - Layer 1: one `LinearClassifier` per EEG channel (23 total), each
        trained independently (via `train_logistic`) on that channel's own
        7 Poincaré features to predict seizure/background for that channel
        alone. This mirrors the reference paper's idea of first getting a
        per-channel "opinion" before combining channels. Supervised in
        both modes.
      - Layer 2, depending on `mode`:
          'svm': a single `LinearClassifier` trained (via `train_linear_svm`)
            on the 23 per-channel binary predictions from Layer 1, learning
            how to weigh/combine them into one seizure/background decision.
          'ocsvm': a `OneClassSVM` trained (via `train_one_class_svm`) only
            on the non-seizure epochs' per-channel scores; at inference any
            epoch outside the learned "normal" region is called a seizure.

    Hyperparameters are the module-level constants at the top of this file.

    Args:
        mode: 'svm' or 'ocsvm' (see `MODES`).
        seed: RNG seed, so every mode is compared on identical data.

    Returns:
        Dict with 'accuracy', 'sensitivity' and 'specificity' on a held-out
        synthetic test set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_channels = 23
    fs = 256  # Hz
    epoch_length = 1  # second
    n_samples = fs * epoch_length

    y_train = torch.tensor([1] * 50 + [0] * 50, dtype=torch.long, device=DEVICE)
    y_test = torch.tensor([1] * 20 + [0] * 20, dtype=torch.long, device=DEVICE)

    print(f"\n===== Mode: {mode} =====")
    print("Extracting features for Layer 1...")
    X_train_all_channels = simulate_epoch_features(len(y_train), n_channels, n_samples)
    X_test_all_channels = simulate_epoch_features(len(y_test), n_channels, n_samples)

    # Layer 1: Train 23 separate per-channel classifiers (LDA replacement)
    print("Training Layer 1 (23 logistic classifiers, replacing LDA)...")
    layer1_models = []
    for ch in range(n_channels):
        model = LinearClassifier(7).to(DEVICE)
        X_ch = X_train_all_channels[:, ch, :]
        model = train_logistic(model, X_ch, y_train, epochs=LAYER1_EPOCHS, lr=LAYER1_LR)
        layer1_models.append(model)

    train_l1 = layer1_outputs(layer1_models, X_train_all_channels, mode)
    test_l1 = layer1_outputs(layer1_models, X_test_all_channels, mode)

    if mode == 'svm':
        # Layer 2: Train linear SVM (hinge loss) on the binary outputs of Layer 1
        print("Training Layer 2 (linear SVM, PyTorch hinge loss)...")
        layer2 = LinearClassifier(n_channels).to(DEVICE)
        layer2 = train_linear_svm(
            layer2, train_l1, y_train, C=LAYER2_C,
            epochs=LAYER2_EPOCHS, lr=LAYER2_LR
        )
        y_pred = predict_labels(layer2, test_l1)
    else:
        # Layer 2: Train One-Class SVM on the normal (non-seizure) epochs only
        print("Training Layer 2 (One-Class SVM, normal epochs only)...")
        layer2 = OneClassSVM(n_channels).to(DEVICE)
        layer2 = train_one_class_svm(
            layer2, train_l1[y_train == 0], nu=OCSVM_NU,
            epochs=OCSVM_EPOCHS, lr=OCSVM_LR
        )
        train_outliers = predict_anomalies(layer2, train_l1[y_train == 0]).float().mean().item()
        print(f"Normal training epochs flagged as anomalies: {train_outliers:.2f} (nu={OCSVM_NU})")
        y_pred = predict_anomalies(layer2, test_l1)

    print("Pipeline trained successfully!")

    acc, sen, spe = compute_metrics(y_test, y_pred)
    print(f"accuracy={acc:.2f}  sensitivity={sen:.2f}  specificity={spe:.2f}")
    return {'accuracy': acc, 'sensitivity': sen, 'specificity': spe}


if __name__ == "__main__":
    results = {mode: run_pipeline(mode) for mode in MODES}

    print("\n===== Comparison =====")
    print(f"{'mode':<8}{'sensitivity':>13}{'specificity':>13}{'accuracy':>10}")
    for mode, m in results.items():
        print(f"{mode:<8}{m['sensitivity']:>13.2f}{m['specificity']:>13.2f}{m['accuracy']:>10.2f}")
