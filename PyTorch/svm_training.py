import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caminhos resolvidos a partir da localização do script (não do cwd), para
# que main_pipeline.py funcione tanto rodando de dentro de PyTorch/ quanto
# da raiz do repositório.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# https://en.wikipedia.org/wiki/Support_vector_machine
#     Hinge loss is defined as: max(0, 1 - y * (w^T x + b)), where y in {-1, 1}.
#     The primal objective is: 0.5 * ||w||^2 + C * mean(hinge_loss), where C is the regularization parameter.
class LinearSVM(nn.Module):
    """
    A linear Support Vector Machine, implemented as a single linear layer
    (this is literally all a linear SVM is: a weight vector w, a bias b, and
    a decision score w.x + b). What makes it an "SVM" rather than plain
    linear regression is how it's trained - see `train_linear_svm`, which
    optimizes the hinge-loss objective instead of e.g. mean squared error.
    """

    def __init__(self, n_features):
        """
        Args:
            n_features: number of input features per sample (must match the
                width of the X tensors this model will be called on).
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        """
        Computes the raw decision score for a batch of samples (not a
        probability - just the signed distance-like score w.x + b; positive
        means "seizure" side of the boundary, negative means "background").

        Args:
            x: torch.Tensor of shape (batch_size, n_features).

        Returns:
            torch.Tensor of shape (batch_size,) with one decision score per
            sample.
        """
        return self.linear(x).squeeze(-1)


def fit_scaler(X):
    """
    Computes the per-feature mean and standard deviation of a training set,
    to later standardize (z-score) both train and test data the same way
    (torch replacement for sklearn.preprocessing.StandardScaler().fit()).
    Standardizing matters a lot for an SVM: since the objective penalizes
    ||w||^2, features on a much larger numeric scale would dominate the
    decision boundary regardless of how informative they actually are.

    Args:
        X: torch.Tensor of shape (n_samples, n_features) - the TRAINING
            data only (never fit this on test data, or information from the
            test set leaks into training).

    Returns:
        Tuple (mean, std): both are 1D torch.Tensors of shape (n_features,).
        Any feature with zero variance (constant value) gets std=1 instead
        of 0, so dividing by it later doesn't produce NaN/Inf.
    """
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def apply_scaler(X, mean, std):
    """
    Standardizes data using a previously fitted (mean, std) pair: (X - mean)
    / std. Apply the SAME mean/std (fitted on the training set via
    `fit_scaler`) to both training and test/validation data, so all splits
    live on the same numeric scale.

    Args:
        X: torch.Tensor of shape (n_samples, n_features) to standardize.
        mean: 1D torch.Tensor of shape (n_features,), from `fit_scaler`.
        std: 1D torch.Tensor of shape (n_features,), from `fit_scaler`.

    Returns:
        torch.Tensor of the same shape as X, standardized feature-wise.
    """
    return (X - mean) / std


def stratified_kfold_indices(y, n_splits=3, seed=42):
    """
    Splits a dataset into `n_splits` folds for cross-validation, keeping the
    proportion of positive (seizure) vs. negative (background) samples
    roughly the same in every fold (torch/numpy replacement for
    sklearn.model_selection.StratifiedKFold). "Stratified" matters here
    specifically because seizure windows are rare: a plain random split
    could easily produce a fold with zero seizure samples, which would make
    sensitivity undefined for that fold.

    How: shuffles the positive-label indices and negative-label indices
    separately (each with the same `seed`, for reproducibility), splits each
    group into `n_splits` roughly-equal chunks with `np.array_split`, then
    for fold k, uses chunk k of both groups as the validation set and all
    the other chunks (of both groups) as the training set.

    Args:
        y: torch.Tensor of shape (n_samples,) with binary labels (0/1).
        n_splits: number of folds to create.
        seed: random seed controlling the shuffle, for reproducibility.

    Returns:
        List of `n_splits` tuples (train_idx, val_idx), where each is a 1D
        numpy array of integer indices into `y` (and the matching X) for
        that fold's training and validation subsets.
    """
    y_np = y.cpu().numpy()
    rng = np.random.RandomState(seed)
    idx_pos = rng.permutation(np.where(y_np == 1)[0])
    idx_neg = rng.permutation(np.where(y_np == 0)[0])
    pos_folds = np.array_split(idx_pos, n_splits)
    neg_folds = np.array_split(idx_neg, n_splits)

    folds = []
    for k in range(n_splits):
        test_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_idx = np.concatenate(
            [f for i, f in enumerate(pos_folds) if i != k]
            + [f for i, f in enumerate(neg_folds) if i != k]
        )
        folds.append((train_idx, test_idx))
    return folds


def train_linear_svm(X, y, C, epochs=300, lr=0.05, seed=42, class_weight_balanced=True):
    """
    Trains a `LinearSVM` from scratch via gradient descent on the standard
    soft-margin primal SVM objective:
        loss = 0.5 * ||w||^2 + C * mean(sample_weight * hinge_loss)
    where hinge_loss = max(0, 1 - y_signed * decision_score) and
    y_signed in {-1, +1}. This is the same objective sklearn's
    SVC(kernel='linear') solves via quadratic programming - here it's solved
    instead with an ordinary PyTorch optimizer (Adam), which is what makes
    every piece of it (epochs, learning rate, optimizer choice, class
    weighting) directly tunable, unlike a black-box QP solver.

    Intuition for the two loss terms:
      - 0.5*||w||^2 is the regularizer: it pushes the weights toward zero,
        i.e. toward a wider/simpler margin. This is what `C` trades off
        against.
      - hinge_loss is 0 for a point already correctly classified with
        enough margin (y_signed * score >= 1), and grows linearly with how
        far a point is on the wrong side of the margin otherwise - so only
        "hard" points near/past the boundary contribute gradient.
      - C controls the trade-off: large C = fit the training data harder
        (less regularization, higher risk of overfitting); small C = prefer
        a simpler/wider-margin boundary (more regularization, higher risk of
        underfitting).

    Args:
        X: torch.Tensor of shape (n_samples, n_features), already
            standardized (see `apply_scaler`).
        y: torch.Tensor of shape (n_samples,) with binary labels (0/1).
        C: regularization strength (inverse of how much the margin term is
            allowed to dominate) - see the trade-off explanation above.
        epochs: number of full-batch gradient descent steps to run.
        lr: learning rate for the Adam optimizer.
        seed: random seed for weight initialization, for reproducibility.
        class_weight_balanced: if True, re-weights the hinge loss so
            positive (seizure) and negative (background) samples contribute
            equally in total, regardless of how imbalanced the class counts
            are (mirrors sklearn's class_weight='balanced'). If False, every
            sample counts equally, which usually biases the model toward
            always predicting the majority class when seizures are rare.

    Returns:
        A trained `LinearSVM` instance (already in `.eval()` mode).
    """
    torch.manual_seed(seed)
    n_features = X.shape[1]
    model = LinearSVM(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y_signed = (y.float() * 2 - 1)

    if class_weight_balanced:
        n_pos = (y == 1).sum().item()
        n_neg = (y == 0).sum().item()
        total = n_pos + n_neg
        w_pos = total / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = total / (2.0 * n_neg) if n_neg > 0 else 1.0
        sample_weight = torch.where(
            y == 1,
            torch.full_like(y_signed, w_pos),
            torch.full_like(y_signed, w_neg),
        )
    else:
        sample_weight = torch.ones_like(y_signed)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        hinge = torch.clamp(1 - y_signed * out, min=0)
        weighted_hinge = (hinge * sample_weight).mean()
        w = model.linear.weight
        loss = 0.5 * (w ** 2).sum() + C * weighted_hinge
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_svm(model, X):
    """
    Predicts binary labels for a batch of (already standardized) samples,
    by thresholding the model's raw decision score at zero: score > 0 is
    predicted as the positive/seizure class, score <= 0 as the negative/
    background class. Uses `torch.no_grad()` since this is inference-only
    and doesn't need gradients.

    Args:
        model: a trained `LinearSVM` (or anything with the same `.forward`
            signature).
        X: torch.Tensor of shape (n_samples, n_features), standardized the
            same way the model was trained (via `apply_scaler`).

    Returns:
        torch.Tensor of shape (n_samples,), dtype long, with 0/1 predictions.
    """
    with torch.no_grad():
        out = model(X)
    return (out > 0).long()


def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, sensitivity, and specificity from true vs. predicted
    binary labels (torch replacement for sklearn's accuracy_score/
    recall_score, computed here directly from the confusion matrix counts).

    Definitions (with 1 = seizure/positive, 0 = background/negative):
      - accuracy: fraction of all predictions that were correct.
      - sensitivity (a.k.a. recall on the positive class): of all the
        actual seizures, what fraction did the model catch? This is
        usually the metric that matters most for seizure detection, since
        missing a seizure (false negative) is worse than a false alarm.
      - specificity (recall on the negative class): of all the actual
        background windows, what fraction did the model correctly leave
        alone? Low specificity means lots of false alarms.

    Args:
        y_true: torch.Tensor of shape (n_samples,) with the ground-truth 0/1
            labels.
        y_pred: torch.Tensor of shape (n_samples,) with the predicted 0/1
            labels (e.g. from `predict_svm`).

    Returns:
        Tuple (acc, sen, spe) of plain Python floats. `sen` is NaN if there
        are no positive samples in y_true (sensitivity undefined), and `spe`
        is NaN if there are no negative samples (specificity undefined).
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sen = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spe = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    return acc, sen, spe


def train_patient_specific_models(training_rate=0.50, C_grid=(0.001, 0.01, 0.1, 1, 10, 100),
                                   epochs=300, lr=0.05, n_splits=3, seed=42):
    """
    End-to-end training pipeline: for every patient with a saved feature
    dataset (X_chbXX.pt / y_chbXX.pt, produced by poincare_features.py or
    label_dataset_v2.py), trains one linear SVM per patient, picks its best
    regularization strength C via cross-validation, evaluates it on held-out
    data, and saves the model, scaler, and diagnostic plots.

    How, per patient:
      1. Split the patient's windows chronologically (first `training_rate`
         fraction = train, rest = test) - NOT shuffled, so the test set is
         always in the "future" relative to training, which is the
         realistic scenario for a system meant to detect new seizures.
      2. Fit a scaler (`fit_scaler`) on the training split only, and apply
         it to both splits (`apply_scaler`).
      3. For every candidate C in `C_grid`, run `n_splits`-fold
         cross-validation on the training split (`stratified_kfold_indices`
         + `train_linear_svm` + `compute_metrics`), scoring each fold by
         validation sensitivity, and average across folds. This produces
         one (train_score, val_score) pair per C, which is exactly what
         gets plotted in the "validation curve" - see module docstring
         guidance on reading over/underfitting from it.
      4. Pick the C with the best mean validation sensitivity, then retrain
         a fresh model on the FULL training split with that C (the
         cross-validation models themselves are discarded - they only exist
         to pick C).
      5. Evaluate that final model on the untouched test split
         (`compute_metrics`), and save the model + scaler to
         `models_torch/`, plus the validation-curve plot.
      6. After all patients are processed, saves an aggregate bar chart +
         boxplot comparing sensitivity/specificity/accuracy across every
         patient.

    Args:
        training_rate: fraction of each patient's windows (in chronological
            order) used for training; the rest is held out as the test set.
        C_grid: iterable of candidate regularization strengths to try per
            patient - see `train_linear_svm` for what C controls.
        epochs: number of gradient-descent steps used to train every model
            (both the cross-validation models and the final refit).
        lr: learning rate passed to `train_linear_svm`.
        n_splits: number of cross-validation folds used to score each C.
        seed: random seed for the fold splits and model initialization.

    Returns:
        None. Side effects: prints per-patient and aggregate metrics to
        stdout; writes `models_torch/svm_linear_<patient>.pt` and
        `models_torch/scaler_<patient>.pt` per patient; writes
        `results/curves_torch/validation_curve_<patient>.png` per patient;
        writes `results/svm_linear_performance_torch.png` once at the end.
    """
    print(f"Iniciando Treinamento Específico por Paciente (Taxa de Treino: {training_rate*100}%)")
    print("Modelo: SVM Linear via PyTorch (hinge loss, gradiente descendente) - controle total dos hiperparâmetros")
    print(f"Hiperparâmetros: epochs={epochs}, lr={lr}, C_grid={C_grid}, n_splits={n_splits}, device={DEVICE}")

    all_accuracies = []
    all_sensitivities = []
    all_specificities = []
    trained_patients = []

    curves_dir = os.path.join(SCRIPT_DIR, 'results', 'curves_torch')
    models_dir = os.path.join(SCRIPT_DIR, 'models_torch')
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        x_path = os.path.join(SCRIPT_DIR, f"X_{patient_id}.pt")
        y_path = os.path.join(SCRIPT_DIR, f"y_{patient_id}.pt")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue

        print(f"\n{'-'*40}")
        print(f"Processando Paciente: {patient_id}")

        X = torch.load(x_path).to(DEVICE).float()
        y = torch.load(y_path).to(DEVICE).long()

        # 1. Divisão Cronológica (Sem embaralhamento)
        split_idx = int(len(X) * training_rate)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if y_train.sum().item() == 0 or y_test.sum().item() == 0:
            print(f"[AVISO] {patient_id} sem crises no treino ou teste. Pulando paciente.")
            continue

        # 2. Padronização
        mean, std = fit_scaler(X_train)
        X_train_scaled = apply_scaler(X_train, mean, std)
        X_test_scaled = apply_scaler(X_test, mean, std)

        # 3. Stratified K-Fold + grid search manual sobre C
        print(f"[{patient_id}] Tunando hiperparâmetro C via {n_splits}-fold CV (scoring=recall)...")
        folds = stratified_kfold_indices(y_train, n_splits=n_splits, seed=seed)

        mean_train_scores, mean_test_scores = [], []
        for C in C_grid:
            fold_train_scores, fold_val_scores = [], []
            for train_idx, val_idx in folds:
                Xf_train = X_train_scaled[train_idx]
                yf_train = y_train[train_idx]
                Xf_val = X_train_scaled[val_idx]
                yf_val = y_train[val_idx]

                if yf_train.sum().item() == 0 or yf_val.sum().item() == 0:
                    continue

                fold_model = train_linear_svm(Xf_train, yf_train, C=C, epochs=epochs, lr=lr, seed=seed)

                _, train_sen, _ = compute_metrics(yf_train, predict_svm(fold_model, Xf_train))
                _, val_sen, _ = compute_metrics(yf_val, predict_svm(fold_model, Xf_val))
                fold_train_scores.append(train_sen)
                fold_val_scores.append(val_sen)

            mean_train_scores.append(np.nanmean(fold_train_scores) if fold_train_scores else np.nan)
            mean_test_scores.append(np.nanmean(fold_val_scores) if fold_val_scores else np.nan)

        best_idx = int(np.nanargmax(mean_test_scores))
        best_C = C_grid[best_idx]

        # 4. Refit no treino completo com o melhor C
        model = train_linear_svm(X_train_scaled, y_train, C=best_C, epochs=epochs, lr=lr, seed=seed)

        # 5. Avaliação no conjunto de teste (futuro cronológico)
        y_pred = predict_svm(model, X_test_scaled)
        acc, sen, spe = compute_metrics(y_test, y_pred)

        all_accuracies.append(acc)
        all_sensitivities.append(sen)
        all_specificities.append(spe)
        trained_patients.append(patient_id)

        print(f"[{patient_id}] Melhor Parâmetro: C={best_C}")
        print(f"[{patient_id}] Sensibilidade: {sen:.4f} | Especificidade: {spe:.4f} | Acurácia: {acc:.4f}")

        # Salva os modelos
        torch.save(
            {'state_dict': model.state_dict(), 'n_features': X_train.shape[1], 'best_C': best_C},
            os.path.join(models_dir, f'svm_linear_{patient_id}.pt')
        )
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, os.path.join(models_dir, f'scaler_{patient_id}.pt'))

        # 6. PLOT DA CURVA DE VALIDAÇÃO DO PARÂMETRO 'C'
        plt.figure(figsize=(10, 6))
        plt.title(f'Curva de Validação SVM Linear (PyTorch) - Paciente {patient_id}')
        plt.xlabel('Parâmetro C (Regularização)')
        plt.ylabel('Score (Recall/Sensibilidade)')

        plt.semilogx(C_grid, mean_train_scores, label='Score de Treino', color='blue', marker='o')
        plt.semilogx(C_grid, mean_test_scores, label='Score de Validação (CV)', color='orange', marker='s')

        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')
        plt.tight_layout()

        plt.savefig(os.path.join(curves_dir, f'validation_curve_{patient_id}.png'), dpi=300)
        plt.close()

    # 7. Relatório Final e Gráfico Geral
    if all_accuracies:
        print(f"\n{'='*40}")
        print("PERFORMANCE GERAL DO SVM LINEAR (PyTorch)")
        print(f"{'='*40}")
        print(f"Média Sensibilidade: {np.mean(all_sensitivities):.4f}")
        print(f"Média Especificidade: {np.mean(all_specificities):.4f}")
        print(f"Média Acurácia:      {np.mean(all_accuracies):.4f}")
        print(f"{'='*40}")

        print("\nGerando gráficos de performance geral...")

        patient_labels = trained_patients
        x = np.arange(len(patient_labels))
        width = 0.25

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        ax1.bar(x - width, all_sensitivities, width, label='Sensibilidade', color='#1f77b4')
        ax1.bar(x, all_specificities, width, label='Especificidade', color='#ff7f0e')
        ax1.bar(x + width, all_accuracies, width, label='Acurácia', color='#2ca02c')

        ax1.set_ylabel('Score')
        ax1.set_title('Performance SVM Linear (PyTorch) por Paciente')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patient_labels, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim([0, 1.05])

        data = [all_sensitivities, all_specificities, all_accuracies]
        ax2.boxplot(data, labels=['Sensibilidade', 'Especificidade', 'Acurácia'], patch_artist=True)
        ax2.set_title('Distribuição Geral de Performance')
        ax2.set_ylabel('Score')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()

        plt.savefig(os.path.join(SCRIPT_DIR, 'results', 'svm_linear_performance_torch.png'), dpi=300)
        print(f"Gráficos salvos na pasta '{os.path.join(SCRIPT_DIR, 'results')}'")


if __name__ == "__main__":
    train_patient_specific_models(training_rate=0.50)
