import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearSVM(nn.Module):
    """Linear SVM implemented as a single linear layer trained with hinge loss."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def fit_scaler(X):
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std


def stratified_kfold_indices(y, n_splits=3, seed=42):
    """Manual stratified K-Fold (replaces sklearn.model_selection.StratifiedKFold)."""
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
    Trains a linear SVM via gradient descent on the standard soft-margin
    primal objective: 0.5*||w||^2 + C * mean(hinge_loss), with optional
    class-balanced sample weights (mirrors sklearn's class_weight='balanced').
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
    with torch.no_grad():
        out = model(X)
    return (out > 0).long()


def compute_metrics(y_true, y_pred):
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
    print(f"Iniciando Treinamento Específico por Paciente (Taxa de Treino: {training_rate*100}%)")
    print("Modelo: SVM Linear via PyTorch (hinge loss, gradiente descendente) - controle total dos hiperparâmetros")
    print(f"Hiperparâmetros: epochs={epochs}, lr={lr}, C_grid={C_grid}, n_splits={n_splits}, device={DEVICE}")

    all_accuracies = []
    all_sensitivities = []
    all_specificities = []
    trained_patients = []

    os.makedirs('results/curves_torch', exist_ok=True)
    os.makedirs('models_torch', exist_ok=True)

    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        x_path = f"X_{patient_id}.pt"
        y_path = f"y_{patient_id}.pt"

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
            f'models_torch/svm_linear_{patient_id}.pt'
        )
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, f'models_torch/scaler_{patient_id}.pt')

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

        plt.savefig(f'results/curves_torch/validation_curve_{patient_id}.png', dpi=300)
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

        plt.savefig('results/svm_linear_performance_torch.png', dpi=300)
        print("Gráficos salvos na pasta 'results/'")


if __name__ == "__main__":
    train_patient_specific_models(training_rate=0.50)
