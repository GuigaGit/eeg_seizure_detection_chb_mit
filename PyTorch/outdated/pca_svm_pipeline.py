import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from joblib import Parallel, delayed

from global_parse_dataset import extrair_dados_sumario
from poincare_features import process_single_file, CHANNELS_TO_KEEP, SCRIPT_DIR, DATASET_DIR
from svm_training import fit_scaler, apply_scaler, compute_metrics

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Real data (CHB-MIT)
# ==========================================
# DATASET_DIR is shared with the rest of the pipeline (poincare_features.py):
# the folder with one subfolder per patient (chb01/, chb02/, ...), set with
#   export CHB_MIT_DATASET_DIR=/caminho/para/dataset_chbmit
# Using the same setting guarantees the cached X_chbXX.pt features and this
# script's labels come from the same dataset.
WINDOW_SEC = 1           # epoch length, in seconds
SFREQ = 256              # CHB-MIT sampling frequency, in Hz
N_CHANNELS = len(CHANNELS_TO_KEEP)
N_FEATURES = 7           # Poincaré features per channel
TRAINING_RATE = 0.50     # chronological train/test split per patient (same as svm_training.py)
# Reuse X_chbXX.pt / y_chbXX.pt saved by poincare_features.py (same features,
# same format). Set False to always re-extract them from the .edf files.
USE_FEATURE_CACHE = True

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
# Real Data Loading
# ==========================================
def load_patient_labels(patient_id, base_path=DATASET_DIR):
    """
    Reads the seizure annotations of one patient straight from its
    `chbXX-summary.txt` in `base_path` (via
    `global_parse_dataset.extrair_dados_sumario`), so the labels always
    match the dataset folder being read.

    Args:
        patient_id: e.g. "chb01".
        base_path: root folder with one subfolder per patient.

    Returns:
        pandas DataFrame with columns 'file_name', 'start_sec', 'end_sec',
        'label', 'patient' (the format `process_single_file` expects), or
        None if the summary file doesn't exist.
    """
    summary_path = os.path.join(base_path, patient_id, f'{patient_id}-summary.txt')
    if not os.path.exists(summary_path):
        return None
    df = pd.DataFrame(extrair_dados_sumario(summary_path))
    df['patient'] = patient_id
    return df


def load_patient_features(patient_id, base_path=DATASET_DIR, use_cache=USE_FEATURE_CACHE):
    """
    Loads one patient's labeled Poincaré-feature windows from the real
    CHB-MIT recordings.

    How: if `use_cache` and `X_<patient>.pt` / `y_<patient>.pt` already
    exist next to this script (saved by poincare_features.py or by a
    previous run of this script), loads them. Otherwise, runs
    `poincare_features.process_single_file` on every .edf of the patient in
    parallel - seizure windows (label 1) plus 15 random background windows
    per file (label 0) - and saves the result to that same cache.

    Args:
        patient_id: e.g. "chb01".
        base_path: root folder with one subfolder per patient.
        use_cache: whether to reuse previously extracted features.

    Returns:
        Tuple (X, y): X is a torch.Tensor of shape (n_windows, N_CHANNELS,
        N_FEATURES), y a torch.Tensor of shape (n_windows,) with 0/1 labels,
        both in recording order. None if the patient has no data.
    """
    x_path = os.path.join(SCRIPT_DIR, f'X_{patient_id}.pt')
    y_path = os.path.join(SCRIPT_DIR, f'y_{patient_id}.pt')

    if use_cache and os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading cached features: {x_path}")
        X = torch.load(x_path, map_location=DEVICE)
        y = torch.load(y_path, map_location=DEVICE)
    else:
        labels = load_patient_labels(patient_id, base_path)
        if labels is None:
            return None
        print(f"Extracting features from {os.path.join(base_path, patient_id)} ...")
        results = Parallel(n_jobs=-1)(
            delayed(process_single_file)(f, file_group, base_path, WINDOW_SEC, SFREQ)
            for f, file_group in labels.groupby('file_name')
        )
        results = [r for r in results if r is not None and r[0].shape[0] > 0]
        if not results:
            return None
        X = torch.cat([r[0].to(DEVICE) for r in results], dim=0)
        y = torch.cat([r[1].to(DEVICE) for r in results], dim=0)
        torch.save(X, x_path)
        torch.save(y, y_path)

    # label_dataset_v2.py writes different features to the same X_chbXX.pt files
    if X.shape[1] != N_CHANNELS * N_FEATURES:
        raise ValueError(
            f"{x_path} has {X.shape[1]} features per window, expected "
            f"{N_CHANNELS * N_FEATURES} Poincaré features ({N_CHANNELS} channels x {N_FEATURES}). "
            f"Re-run poincare_features.py or set USE_FEATURE_CACHE = False."
        )

    # Flat (n, 7 * channels) -> (n, channels, 7): channel `ch` owns [7*ch : 7*(ch+1)]
    X = X.float().reshape(-1, N_CHANNELS, N_FEATURES)
    return X, y.long()


# ==========================================
# Main Pipeline & Classification
# ==========================================

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


def run_pipeline(X_train_all_channels, y_train, X_test_all_channels, y_test, mode='ocsvm', seed=42):
    """
    Trains the full 2-layer classification architecture on one patient's
    (standardized) training windows and evaluates it on the held-out ones.

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
        X_train_all_channels: torch.Tensor (n_train, n_channels, 7).
        y_train: torch.Tensor (n_train,) with 0/1 labels.
        X_test_all_channels: torch.Tensor (n_test, n_channels, 7).
        y_test: torch.Tensor (n_test,) with 0/1 labels.
        mode: 'svm' or 'ocsvm' (see `MODES`).
        seed: RNG seed, so every mode starts from the same initialization.

    Returns:
        Dict with 'accuracy', 'sensitivity' and 'specificity' on the test set.
    """
    torch.manual_seed(seed)
    n_channels = X_train_all_channels.shape[1]

    # Layer 1: Train 23 separate per-channel classifiers (LDA replacement)
    layer1_models = []
    for ch in range(n_channels):
        model = LinearClassifier(N_FEATURES).to(DEVICE)
        X_ch = X_train_all_channels[:, ch, :]
        model = train_logistic(model, X_ch, y_train, epochs=LAYER1_EPOCHS, lr=LAYER1_LR)
        layer1_models.append(model)

    train_l1 = layer1_outputs(layer1_models, X_train_all_channels, mode)
    test_l1 = layer1_outputs(layer1_models, X_test_all_channels, mode)

    if mode == 'svm':
        # Layer 2: Train linear SVM (hinge loss) on the binary outputs of Layer 1
        layer2 = LinearClassifier(n_channels).to(DEVICE)
        layer2 = train_linear_svm(
            layer2, train_l1, y_train, C=LAYER2_C,
            epochs=LAYER2_EPOCHS, lr=LAYER2_LR
        )
        y_pred = predict_labels(layer2, test_l1)
    else:
        # Layer 2: Train One-Class SVM on the normal (non-seizure) epochs only
        layer2 = OneClassSVM(n_channels).to(DEVICE)
        layer2 = train_one_class_svm(
            layer2, train_l1[y_train == 0], nu=OCSVM_NU,
            epochs=OCSVM_EPOCHS, lr=OCSVM_LR
        )
        y_pred = predict_anomalies(layer2, test_l1)

    acc, sen, spe = compute_metrics(y_test, y_pred)
    print(f"  [{mode:<5}] accuracy={acc:.2f}  sensitivity={sen:.2f}  specificity={spe:.2f}")
    return {'accuracy': acc, 'sensitivity': sen, 'specificity': spe}


def run_all_patients(base_path=DATASET_DIR, training_rate=TRAINING_RATE, modes=MODES):
    """
    Patient-specific evaluation on the real CHB-MIT data (same protocol as
    `svm_training.train_patient_specific_models`): for each patient chb01..
    chb24, load its windows (`load_patient_features`), split them
    chronologically (first `training_rate` = train, rest = test), standardize
    every channel's features with the training split's mean/std, then train
    and evaluate every Layer 2 mode on that same split.

    Args:
        base_path: root folder with one subfolder per patient.
        training_rate: fraction of each patient's windows used for training.
        modes: Layer 2 modes to evaluate (see `MODES`).

    Returns:
        Dict {mode: {patient_id: metrics dict}}.
    """
    print(f"Dataset: {base_path}")
    if not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {base_path}. "
            f"Set it with: export CHB_MIT_DATASET_DIR=/caminho/para/dataset_chbmit"
        )

    results = {mode: {} for mode in modes}
    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        data = load_patient_features(patient_id, base_path)
        if data is None:
            continue
        X, y = data

        # Chronological split (no shuffling): the test set is always "future" data
        split_idx = int(len(X) * training_rate)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if y_train.sum().item() == 0 or y_test.sum().item() == 0:
            print(f"[AVISO] {patient_id} sem crises no treino ou teste. Pulando paciente.")
            continue
        if (y_train == 0).sum().item() == 0:
            # The One-Class SVM is trained on normal epochs only
            print(f"[AVISO] {patient_id} sem janelas normais no treino. Pulando paciente.")
            continue

        # Standardize each (channel, feature) pair with the training split only - raw
        # features are in µV and differ by orders of magnitude (e.g. energy vs. CoV)
        mean, std = fit_scaler(X_train.reshape(len(X_train), -1))
        X_train = apply_scaler(X_train.reshape(len(X_train), -1), mean, std).reshape(X_train.shape)
        X_test = apply_scaler(X_test.reshape(len(X_test), -1), mean, std).reshape(X_test.shape)

        print(f"{patient_id}: train={len(y_train)} ({int(y_train.sum())} seizure)  "
              f"test={len(y_test)} ({int(y_test.sum())} seizure)")
        for mode in modes:
            results[mode][patient_id] = run_pipeline(X_train, y_train, X_test, y_test, mode=mode)

    return results


if __name__ == "__main__":
    results = run_all_patients()

    print("\n===== Comparison (mean over patients) =====")
    print(f"{'mode':<8}{'patients':>10}{'sensitivity':>13}{'specificity':>13}{'accuracy':>10}")
    for mode, per_patient in results.items():
        if not per_patient:
            print(f"{mode:<8}{0:>10}  (no patient data found)")
            continue
        mean = {k: np.nanmean([m[k] for m in per_patient.values()])
                for k in ('sensitivity', 'specificity', 'accuracy')}
        print(f"{mode:<8}{len(per_patient):>10}{mean['sensitivity']:>13.2f}"
              f"{mean['specificity']:>13.2f}{mean['accuracy']:>10.2f}")
