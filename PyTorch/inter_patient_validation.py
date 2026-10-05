import os
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Save-only script; avoids X11/ICE crashes at exit on headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from svm_training import LinearSVM, apply_scaler, compute_metrics, predict_svm, DEVICE, SCRIPT_DIR


def run_inter_patient_validation():
    """
    Measures how well each patient's individually-trained SVM (from
    `svm_training.py`) generalizes to every OTHER patient's data - a
    "leave-one-patient-out"-style cross-check, useful for answering "is this
    model learning something patient-specific about EEG in general, or did
    it just memorize this one patient's quirks?".

    How: for every trained patient A (a model + scaler exist in
    `models_torch/`), loads A's model and scaler, then for every patient B
    with a saved dataset (including A itself), standardizes B's features
    using A's scaler and evaluates A's model on B's full dataset
    (`compute_metrics`). This produces three N x N matrices (accuracy,
    sensitivity, specificity; N = number of trained patients), where entry
    (i, j) = "model trained on patient i, tested on patient j's data". The
    diagonal (i == j, i.e. a patient tested on their own data) is excluded
    from the summary statistics, since that's the "easy" same-patient case
    already reported by `svm_training.py` - what matters here is the
    off-diagonal, cross-patient performance.

    Takes no arguments and returns nothing; instead it prints a per-patient
    and overall cross-patient report to stdout, and saves two heatmaps
    (mean cross-patient sensitivity and specificity) to
    `results/cross_patient_torch/inter_patient_heatmaps.png`. Requires
    `svm_training.train_patient_specific_models` to have been run first, so
    that `models_torch/` and the per-patient `X_*.pt`/`y_*.pt` files exist;
    prints an error and returns early otherwise.
    """
    print(f"\n{'='*50}")
    print("Iniciando Validação Inter-Pacientes (Cross-Patient) - PyTorch")
    print(f"{'='*50}")

    # 1. Identificar quais pacientes possuem modelos treinados
    models_dir = os.path.join(SCRIPT_DIR, 'models_torch')
    trained_patients = []
    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        model_path = os.path.join(models_dir, f"svm_linear_{patient_id}.pt")
        if os.path.exists(model_path):
            trained_patients.append(patient_id)

    n_patients = len(trained_patients)
    if n_patients == 0:
        print("[ERRO] Nenhum modelo encontrado na pasta 'models_torch/'. Treine os modelos primeiro.")
        return

    print(f"Modelos encontrados: {n_patients} pacientes.\n")

    matrix_acc = np.zeros((n_patients, n_patients))
    matrix_sen = np.zeros((n_patients, n_patients))
    matrix_spe = np.zeros((n_patients, n_patients))

    # 2. Loop Duplo: Treino vs Teste
    for i, train_id in enumerate(trained_patients):
        print(f"Avaliando o modelo do paciente: {train_id} contra os demais...")

        checkpoint = torch.load(os.path.join(models_dir, f"svm_linear_{train_id}.pt"), map_location=DEVICE)
        model = LinearSVM(checkpoint['n_features']).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        scaler = torch.load(os.path.join(models_dir, f"scaler_{train_id}.pt"), map_location=DEVICE)
        mean, std = scaler['mean'].to(DEVICE), scaler['std'].to(DEVICE)

        for j, test_id in enumerate(trained_patients):
            x_path = os.path.join(SCRIPT_DIR, f"X_{test_id}.pt")
            y_path = os.path.join(SCRIPT_DIR, f"y_{test_id}.pt")

            if not os.path.exists(x_path) or not os.path.exists(y_path):
                continue

            X_test_raw = torch.load(x_path).to(DEVICE).float()
            y_test = torch.load(y_path).to(DEVICE).long()

            try:
                X_test_scaled = apply_scaler(X_test_raw, mean, std)
                y_pred = predict_svm(model, X_test_scaled)

                acc, sen, spe = compute_metrics(y_test, y_pred)
                matrix_acc[i, j] = acc
                matrix_sen[i, j] = sen if y_test.sum().item() > 0 else np.nan
                matrix_spe[i, j] = spe if (len(y_test) - y_test.sum().item()) > 0 else np.nan

            except Exception as e:
                print(f"  [ERRO DIMENSIONAL] Falha ao testar {train_id} em {test_id}: {e}")
                matrix_acc[i, j] = np.nan
                matrix_sen[i, j] = np.nan
                matrix_spe[i, j] = np.nan

            finally:
                del X_test_raw
                if 'X_test_scaled' in locals():
                    del X_test_scaled
                gc.collect()

    # 3. Impressão dos Resultados no Terminal
    print(f"\n{'='*50}")
    print("RELATÓRIO DE VALIDAÇÃO CRUZADA INTER-PACIENTES (PyTorch)")
    print(f"{'='*50}")

    np.fill_diagonal(matrix_sen, np.nan)
    np.fill_diagonal(matrix_spe, np.nan)
    np.fill_diagonal(matrix_acc, np.nan)

    print(f"{'Modelo':<10} | {'Acurácia Média':<16} | {'Sensib. Média':<15} | {'Especif. Média':<15}")
    print("-" * 65)

    for i, train_id in enumerate(trained_patients):
        avg_acc = np.nanmean(matrix_acc[i, :])
        avg_sen = np.nanmean(matrix_sen[i, :])
        avg_spe = np.nanmean(matrix_spe[i, :])

        print(f"{train_id:<10} | {avg_acc:.4f}           | {avg_sen:.4f}          | {avg_spe:.4f}")

    print("-" * 65)
    print(f"{'MÉDIA GERAL':<10} | {np.nanmean(matrix_acc):.4f}           | {np.nanmean(matrix_sen):.4f}          | {np.nanmean(matrix_spe):.4f}")
    print(f"{'='*50}\n")

    # 4. Geração dos Mapas de Calor (Heatmaps)
    cross_patient_dir = os.path.join(SCRIPT_DIR, 'results', 'cross_patient_torch')
    print(f"Salvando Heatmaps em '{os.path.join(cross_patient_dir, 'inter_patient_heatmaps.png')}'...")
    os.makedirs(cross_patient_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(matrix_sen, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=trained_patients, yticklabels=trained_patients, ax=ax1, vmin=0, vmax=1)
    ax1.set_title("Sensibilidade Inter-Paciente (Média)")
    ax1.set_xlabel("Testado no Paciente (Dados)")
    ax1.set_ylabel("Treinado no Paciente (Modelo)")

    sns.heatmap(matrix_spe, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=trained_patients, yticklabels=trained_patients, ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Especificidade Inter-Paciente (Média)")
    ax2.set_xlabel("Testado no Paciente (Dados)")
    ax2.set_ylabel("Treinado no Paciente (Modelo)")

    plt.tight_layout()
    plt.savefig(os.path.join(cross_patient_dir, 'inter_patient_heatmaps.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_inter_patient_validation()
