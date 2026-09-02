import os
import numpy as np
import pandas as pd
import mne
import torch
import matplotlib.pyplot as plt
from poincare_features import time_delay_embedding, get_poincare_intersections, pca_transform, fit_line, SCRIPT_DIR, REPO_ROOT, DATASET_DIR

# Desativar logs excessivos do MNE
mne.set_log_level('ERROR')


def load_real_segments(base_path=None):
    """
    Loads two real 1-second EEG segments from patient chb01 - one from
    quiet background activity and one from the very start of an annotated
    seizure - purely for visual sanity-checking of the Poincaré feature
    pipeline (see `plot_pipeline_validation`), not for training.

    How: reads `chb_mit_global_labels.csv` to find chb01's first file with
    an annotated seizure, loads that .edf with MNE, picks a single
    representative channel ('P7-T7'), and slices out:
      - a "background" window starting 5s into the recording (early enough
        to be unaffected by any startup artifacts, and - by construction of
        this specific file - well before the seizure);
      - a "seizure" window starting at t=3026s, a timestamp hardcoded for
        this specific chb01 recording's known seizure onset.

    Args:
        base_path: root folder containing one subfolder per patient;
            defaults to `DATASET_DIR` (the configured CHB-MIT dataset
            location) if not given.

    Returns:
        Tuple (bg_signal, sz_signal, sfreq): `bg_signal` and `sz_signal` are
        1D numpy arrays of raw voltage samples (1 second each), `sfreq` is
        the recording's sampling frequency in Hz.
    """
    base_path = base_path or DATASET_DIR
    df = pd.read_csv(os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv'))
    df_patient = df[df['patient'] == 'chb01']

    seizure_file_info = df_patient[df_patient['label'] == 1].iloc[0]
    file_name = seizure_file_info['file_name']

    edf_path = os.path.join(base_path, 'chb01', file_name)
    print(f"Carregando arquivo EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    channel_name = 'P7-T7'
    raw.pick_channels([channel_name])
    data, times = raw[:, :]
    signal = data[0]
    sfreq = int(raw.info['sfreq'])

    # 1 segundo de sinal = 256 amostras
    bg_start_idx = int(5 * sfreq)
    bg_signal = signal[bg_start_idx: bg_start_idx + sfreq]

    sz_start_idx = int(3026 * sfreq)
    sz_signal = signal[sz_start_idx: sz_start_idx + sfreq]

    return bg_signal, sz_signal, sfreq


def plot_pipeline_validation(signal, title_prefix="Segmento"):
    """
    Draws a 4-panel diagnostic figure that visually walks through every
    stage of the Poincaré feature pipeline for one 1-second signal, so you
    can sanity-check the math step by step instead of trusting it blindly:
        raw signal -> phase-space embedding -> Poincaré section -> discrete
        intersection sequence.

    The 4 panels:
      1. The raw EEG signal in the time domain.
      2. The reconstructed phase-space trajectory (`time_delay_embedding`),
         projected to its top 3 principal components (`pca_transform`) and
         shown as a 3D curve - this is what "reconstructing the dynamics"
         actually looks like geometrically.
      3. The 2D projection (PC1 vs PC2) of that same trajectory, with the
         fitted Poincaré section line (`fit_line`) and the points where the
         trajectory crosses it highlighted - the geometric core of
         `get_poincare_intersections`.
      4. The same crossing points, but plotted as a discrete sequence in the
         order they occur - this is literally the `intersections` array that
         `extract_features` later summarizes into 7 numbers.

    Args:
        signal: 1D array-like with one channel's raw samples for a single
            window (e.g. one second at 256Hz).
        title_prefix: label used in the figure's title and in the saved
            file's name (e.g. "Segmento Normal (Background)"), to tell
            multiple calls' outputs apart.

    Returns:
        None. Side effects: saves the figure to
        `results/pipeline_validation_torch/<title_prefix>_validation.png`
        and displays it interactively via `plt.show()`.
    """
    # 1. Reconstrução do Espaço de Fase 5D
    embedded = time_delay_embedding(signal, d=5, tau=6)

    # 2. PCA para redução de dimensionalidade (5D -> todos os componentes)
    coords_reduced = pca_transform(embedded).cpu().numpy()

    pc1 = coords_reduced[:, 0]
    pc2 = coords_reduced[:, 1]
    pc3 = coords_reduced[:, 2] if coords_reduced.shape[1] > 2 else np.zeros_like(pc1)

    # Ajuste polinomial de grau 1 (a Seção de Poincaré)
    m, c = fit_line(torch.as_tensor(pc1), torch.as_tensor(pc2))
    m, c = m.item(), c.item()

    x_line = np.linspace(min(pc1), max(pc1), 100)
    y_line = m * x_line + c

    # 3. Encontrar interseções
    intersections_pc1 = get_poincare_intersections(embedded).cpu().numpy()

    # --- Configuração do Plot (Grade 2x2) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Validação Completa do Pipeline (PyTorch) - {title_prefix}", fontsize=16, fontweight='bold')

    # Subplot [0, 0]: Sinal de EEG Bruto
    axs[0, 0].plot(np.arange(len(signal)) / 256.0, signal, color='darkblue', lw=1.5)
    axs[0, 0].set_title("1. Sinal de EEG de 1 segundo")
    axs[0, 0].set_xlabel("Tempo (s)")
    axs[0, 0].set_ylabel("Amplitude (uV)")
    axs[0, 0].grid(True, alpha=0.5)

    # Subplot [0, 1]: Trajetória Espaço de Fase 3D
    pos = axs[0, 1].get_position()
    axs[0, 1].remove()
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_3d.set_position(pos)
    ax_3d.plot3D(pc1, pc2, pc3, color='purple', alpha=0.8, lw=1)
    ax_3d.scatter3D(pc1[0], pc2[0], pc3[0], color='green', s=40, label='Início')
    ax_3d.scatter3D(pc1[-1], pc2[-1], pc3[-1], color='red', s=40, label='Fim')
    ax_3d.set_title("2. Trajetória Reduzida (PCs 1, 2, 3)")
    ax_3d.set_xlabel("PC 1")
    ax_3d.set_ylabel("PC 2")
    ax_3d.set_zlabel("PC 3")
    ax_3d.legend()

    # Subplot [1, 0]: Seção de Poincaré 2D e Pontos de Cruzamento
    axs[1, 0].plot(pc1, pc2, color='gray', alpha=0.6, linestyle='--', label='Trajetória 2D')
    axs[1, 0].plot(x_line, y_line, color='red', lw=2, label='Seção de Poincaré (Linha)')
    intersections_pc2 = m * intersections_pc1 + c
    axs[1, 0].scatter(intersections_pc1, intersections_pc2, color='gold', edgecolor='black', s=50, zorder=5, label='Interseções')
    axs[1, 0].set_title("3. Seção de Poincaré e Cruzamentos")
    axs[1, 0].set_xlabel("PC 1")
    axs[1, 0].set_ylabel("PC 2")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # Subplot [1, 1]: Sequência Temporal Discreta das Interseções
    indices_intersecoes = np.arange(1, len(intersections_pc1) + 1)
    axs[1, 1].plot(indices_intersecoes, intersections_pc1, color='teal', linestyle='-', marker='o',
                  markerfacecolor='gold', markeredgecolor='black', markersize=6, lw=1.5, label='Valor da Interseção ($x_I$)')
    axs[1, 1].set_title("4. Sequência de Interseções de Poincaré")
    axs[1, 1].set_xlabel("Índice do Cruzamento ($n$)")
    axs[1, 1].set_ylabel("Valor Projetado no PC 1 ($x_I(n)$)")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'pipeline_validation_torch')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_validation.png")
    plt.savefig(save_path, dpi=300)
    print(f"Gráfico completo salvo com sucesso em: {save_path}")
    plt.show()


if __name__ == "__main__":
    try:
        normal_segment, seizure_segment, sfreq = load_real_segments()

        plot_pipeline_validation(normal_segment, title_prefix="Segmento Normal (Background)")
        plot_pipeline_validation(seizure_segment, title_prefix="Segmento de Crise (Ictal)")

    except FileNotFoundError as e:
        print(f"\n[AVISO] Não foi possível encontrar os arquivos de dados reais do CHB-MIT: {e}")
