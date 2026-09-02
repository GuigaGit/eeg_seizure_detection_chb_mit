"""
Estima o Maior Expoente de Lyapunov (LLE) de um sinal de EEG usando o método
de Rosenstein (1993) em janelas deslizantes, e anima a evolução desse valor
lado a lado com a forma de onda bruta.

Versão PyTorch: a busca por vizinho mais próximo (scipy.spatial.cKDTree) foi
substituída por uma matriz de distâncias par-a-par via torch.cdist, o que
permite rodar em GPU e encontra o vizinho global mais próximo fora da janela
de Theiler (em vez de buscar apenas entre os top-k candidatos de uma KD-tree).
"""
import os
import argparse
import numpy as np
import pandas as pd
import mne
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from poincare_features import time_delay_embedding, SCRIPT_DIR, REPO_ROOT

mne.set_log_level('ERROR')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Maior Expoente de Lyapunov (Rosenstein)
# ==========================================
def largest_lyapunov_exponent(signal, sfreq, d=5, tau=6, theiler_window=None,
                               trajectory_len=20, fit_range=None):
    """
    Estima o maior expoente de Lyapunov de `signal` pelo método de Rosenstein.

    1. Reconstrói o espaço de fase por time-delay embedding.
    2. Para cada ponto, busca o vizinho mais próximo fora da janela de Theiler
       (evita pares que são apenas vizinhos temporais, não dinâmicos) usando
       uma matriz de distâncias par-a-par (torch.cdist).
    3. Acompanha a divergência log(distância) entre cada par pelos próximos
       `trajectory_len` passos (uma contagem FIXA de amostras, o horizonte de
       curto prazo em que a divergência ainda cresce exponencialmente).
    4. O expoente é a inclinação (ajuste linear) dessa curva de divergência média.

    Retorna o expoente em nats/segundo (log natural). NaN se o segmento
    for curto/plano demais para uma estimativa confiável.
    """
    embedded = time_delay_embedding(signal, d=d, tau=tau)
    n = embedded.shape[0]
    signal_t = torch.as_tensor(signal, dtype=torch.float32, device=DEVICE)
    if n < 20 or signal_t.var().item() <= 1e-12:
        return float('nan')

    if theiler_window is None:
        theiler_window = max(tau * d, int(0.05 * sfreq))

    dist_matrix = torch.cdist(embedded, embedded)
    idx_range = torch.arange(n, device=DEVICE)
    offsets = (idx_range.unsqueeze(1) - idx_range.unsqueeze(0)).abs()
    dist_matrix = dist_matrix.masked_fill(offsets <= theiler_window, float('inf'))

    nearest_idx = torch.argmin(dist_matrix, dim=1)
    nearest_dist = dist_matrix.gather(1, nearest_idx.unsqueeze(1)).squeeze(1)
    valid = torch.isfinite(nearest_dist)

    if valid.sum().item() < 10:
        return float('nan')

    max_horizon = min(trajectory_len, n - 1)
    if max_horizon < 5:
        return float('nan')

    log_div_sum = torch.zeros(max_horizon, device=DEVICE)
    counts = torch.zeros(max_horizon, device=DEVICE)

    for i in torch.nonzero(valid, as_tuple=True)[0].tolist():
        j = int(nearest_idx[i].item())
        span = min(max_horizon, n - max(i, j))
        if span <= 0:
            continue
        diffs = embedded[i:i + span] - embedded[j:j + span]
        dist_k = diffs.norm(dim=1).clamp(min=1e-10)
        log_div_sum[:span] += torch.log(dist_k)
        counts[:span] += 1

    has_data = counts > 0
    if has_data.sum().item() < 5:
        return float('nan')

    mean_log_div = torch.full((max_horizon,), float('nan'), device=DEVICE)
    mean_log_div[has_data] = log_div_sum[has_data] / counts[has_data]

    dt = 1.0 / sfreq
    lo, hi = fit_range if fit_range is not None else (0, max_horizon)
    hi = min(hi, max_horizon)
    fit_mask = has_data.clone()
    fit_mask[:lo] = False
    fit_mask[hi:] = False

    if fit_mask.sum().item() < 3:
        return float('nan')

    k_axis = torch.arange(max_horizon, dtype=torch.float32, device=DEVICE) * dt
    x, y = k_axis[fit_mask], mean_log_div[fit_mask]
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    return solution[0, 0].item()


# ==========================================
# 2. LLE em janelas deslizantes sobre a gravação
# ==========================================
def sliding_lle(signal, sfreq, window_sec=2.0, step_sec=0.5, d=5, tau=6):
    """
    Aplica `largest_lyapunov_exponent` em janelas deslizantes ao longo do sinal.
    Retorna (centros_em_segundos, valores_do_lle).
    """
    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(step_sec * sfreq))

    centers, values = [], []
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        segment = signal[start:start + window_samples]
        lle = largest_lyapunov_exponent(segment, sfreq, d=d, tau=tau)
        centers.append((start + window_samples / 2.0) / sfreq)
        values.append(lle)

    return np.array(centers), np.array(values)


# ==========================================
# 3. Carregamento de dados (formato CHB-MIT)
# ==========================================
def get_seizure_intervals(edf_path, labels_csv=None):
    labels_csv = labels_csv or os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv')
    if not labels_csv or not os.path.exists(labels_csv):
        return []
    df = pd.read_csv(labels_csv)
    file_name = os.path.basename(edf_path)
    rows = df[(df['file_name'] == file_name) & (df['label'] == 1)]
    return list(zip(rows['start_sec'], rows['end_sec']))


def load_channel(edf_path, channel_name, labels_csv=None):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip())

    if channel_name not in raw.ch_names:
        alt = [c for c in raw.ch_names if c.upper().startswith(channel_name.upper())]
        if not alt:
            raise ValueError(
                f"Canal '{channel_name}' não encontrado em {edf_path}. "
                f"Disponíveis: {raw.ch_names}"
            )
        channel_name = alt[0]

    raw.pick_channels([channel_name])
    data, _ = raw[:, :]
    signal = data[0] * 1e6  # V -> uV
    sfreq = raw.info['sfreq']

    return signal, sfreq, channel_name, get_seizure_intervals(edf_path, labels_csv)


def resolve_analysis_window(full_len_samples, sfreq, seizure_intervals, start_sec, duration_sec):
    """
    Decide o trecho [start_idx, end_idx) da gravação a analisar (padrão: 60s
    antes da primeira crise anotada, ou o início da gravação se não houver
    crise) e re-referencia os intervalos de crise para começarem em t=0
    dentro desse trecho.
    """
    if start_sec is None:
        start_sec = max(0.0, seizure_intervals[0][0] - 60) if seizure_intervals else 0.0
    start_idx = int(start_sec * sfreq)
    end_idx = min(full_len_samples, start_idx + int(duration_sec * sfreq))

    local_seizures = [
        (max(0.0, s0 - start_sec), s1 - start_sec)
        for s0, s1 in seizure_intervals
        if s1 > start_sec and s0 < start_sec + duration_sec
    ]
    return start_idx, end_idx, local_seizures


# ==========================================
# 4. Figura de dois painéis (EEG + LLE), mesmo eixo X
# ==========================================
def _build_two_panel_figure(signal, sfreq, seizure_intervals, channel_name, title):
    """
    Monta a figura de dois painéis compartilhando o eixo X (tempo, em segundos):
    o EEG completo em cima e o espaço para a curva de LLE embaixo.
    """
    total_dur = len(signal) / sfreq
    time_axis = np.arange(len(signal)) / sfreq

    fig, (ax_eeg, ax_lle) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax_eeg.plot(time_axis, signal, color='darkblue', lw=0.5)
    ax_eeg.set_ylabel("Amplitude (uV)")
    ax_eeg.set_title(f"Sinal de EEG completo — canal {channel_name}")
    ax_eeg.grid(True, alpha=0.3)
    for s0, s1 in seizure_intervals:
        ax_eeg.axvspan(s0, s1, color='red', alpha=0.15, label='Crise (anotada)')

    ax_lle.set_xlim(0, total_dur)
    ax_lle.set_xlabel("Tempo (s)")
    ax_lle.set_ylabel(r"$\lambda_1$ (nats/s)")
    ax_lle.set_title("Maior Expoente de Lyapunov (janela deslizante)")
    ax_lle.grid(True, alpha=0.3)
    for s0, s1 in seizure_intervals:
        ax_lle.axvspan(s0, s1, color='red', alpha=0.15)

    return fig, ax_eeg, ax_lle, total_dur


def animate_lyapunov(signal, sfreq, centers, lle_values, seizure_intervals,
                      channel_name, save_path=None):
    """
    Anima a evolução do LLE lado a lado com o EEG completo, ambos na mesma
    escala de tempo (eixo X compartilhado).
    """
    fig, ax_eeg, ax_lle, total_dur = _build_two_panel_figure(
        signal, sfreq, seizure_intervals, channel_name,
        f"Expoente de Lyapunov ao vivo — canal {channel_name}")

    cursor_eeg = ax_eeg.axvline(0, color='red', lw=1.5, alpha=0.8)
    cursor_lle = ax_lle.axvline(0, color='red', lw=1.5, alpha=0.8)

    lle_line, = ax_lle.plot([], [], color='teal', lw=1.8)
    finite_vals = lle_values[np.isfinite(lle_values)]
    if finite_vals.size:
        pad = 0.1 * (finite_vals.max() - finite_vals.min() + 1e-6)
        ax_lle.set_ylim(finite_vals.min() - pad, finite_vals.max() + pad)

    value_text = ax_lle.text(
        0.02, 0.92, "", transform=ax_lle.transAxes, fontsize=13,
        fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    def init():
        lle_line.set_data([], [])
        value_text.set_text("")
        return lle_line, cursor_eeg, cursor_lle, value_text

    def update(frame_idx):
        t_now = centers[frame_idx]

        lle_line.set_data(centers[:frame_idx + 1], lle_values[:frame_idx + 1])

        cursor_eeg.set_xdata([t_now, t_now])
        cursor_lle.set_xdata([t_now, t_now])

        current_val = lle_values[frame_idx]
        label = f"t = {t_now:5.1f}s   λ₁ = {current_val:+.3f} nats/s" \
            if np.isfinite(current_val) else f"t = {t_now:5.1f}s   λ₁ = N/D"
        in_seizure = any(s0 <= t_now <= s1 for s0, s1 in seizure_intervals)
        value_text.set_text(("⚠ CRISE — " if in_seizure else "") + label)
        value_text.set_color('red' if in_seizure else 'black')

        return lle_line, cursor_eeg, cursor_lle, value_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(centers), init_func=init,
        interval=80, blit=False, repeat=False
    )

    plt.tight_layout()

    if save_path:
        writer = animation.PillowWriter(fps=12)
        ani.save(save_path, writer=writer)
        print(f"Animação salva em: {save_path}")
    else:
        plt.show()


def plot_final_summary(signal, sfreq, centers, lle_values, seizure_intervals,
                        channel_name, save_path=None):
    """
    Plot estático (sem animação) do sinal completo vs. a evolução completa do
    LLE, na mesma escala de tempo.
    """
    fig, ax_eeg, ax_lle, _ = _build_two_panel_figure(
        signal, sfreq, seizure_intervals, channel_name,
        f"Sinal completo vs. Expoente de Lyapunov — canal {channel_name}")

    ax_lle.plot(centers, lle_values, color='teal', lw=1.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Plot salvo em: {save_path}")
    else:
        plt.show()


# ==========================================
# 5. Processa TODOS os canais do arquivo
# ==========================================
def process_all_channels(edf_path, out_dir=None,
                          labels_csv=None,
                          start_sec=None, duration_sec=180.0,
                          window_sec=2.0, step_sec=0.5):
    """
    Roda a estimativa de LLE em janela deslizante para TODOS os canais do
    arquivo .edf (não apenas um) e salva, para cada canal, o plot final
    estático em `out_dir`.
    """
    out_dir = out_dir or os.path.join(SCRIPT_DIR, 'results', 'lyapunov_torch')

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip())
    sfreq = raw.info['sfreq']
    seizure_intervals = get_seizure_intervals(edf_path, labels_csv)

    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for i, ch_name in enumerate(raw.ch_names):
        print(f"[{i + 1}/{len(raw.ch_names)}] Canal {ch_name}...")
        signal = raw.get_data(picks=ch_name)[0] * 1e6

        start_idx, end_idx, local_seizures = resolve_analysis_window(
            len(signal), sfreq, seizure_intervals, start_sec, duration_sec)
        segment = signal[start_idx:end_idx]

        centers, lle_values = sliding_lle(
            segment, sfreq, window_sec=window_sec, step_sec=step_sec)
        print(f"    {len(centers)} janelas, "
              f"λ₁ médio = {np.nanmean(lle_values):+.3f} nats/s")

        safe_name = ch_name.replace('/', '-')
        save_path = os.path.join(out_dir, f"{base_name}_{safe_name}.png")
        plot_final_summary(segment, sfreq, centers, lle_values, local_seizures,
                            ch_name, save_path=save_path)
        saved_paths.append(save_path)

    return saved_paths


# ==========================================
# 6. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Estima e anima o Expoente de Lyapunov de um sinal de EEG (PyTorch).")
    parser.add_argument('--edf', default=os.path.join(REPO_ROOT, 'dataset_chbmit', 'chb01', 'chb01_03.edf'),
                         help="Caminho do arquivo .edf")
    parser.add_argument('--channel', default='P7-T7',
                         help="Nome do canal a analisar (ignorado com --all-channels).")
    parser.add_argument('--all-channels', action='store_true',
                         help="Processa TODOS os canais do arquivo e salva um plot final "
                              "por canal, em vez de animar um único canal.")
    parser.add_argument('--out-dir', default=os.path.join(SCRIPT_DIR, 'results', 'lyapunov_torch'),
                         help="Pasta onde salvar os plots quando --all-channels for usado.")
    parser.add_argument('--start-sec', type=float, default=None,
                         help="Início do trecho analisado (s). Padrão: 60s antes da crise, ou 0.")
    parser.add_argument('--duration-sec', type=float, default=180.0,
                         help="Duração do trecho analisado, em segundos.")
    parser.add_argument('--window-sec', type=float, default=2.0,
                         help="Tamanho da janela usada em cada estimativa de LLE.")
    parser.add_argument('--step-sec', type=float, default=0.5,
                         help="Passo entre janelas consecutivas.")
    parser.add_argument('--save', default=True,
                         help="Se definido, salva a animação como .gif neste caminho em vez "
                              "de exibi-la (ignorado com --all-channels).")
    args = parser.parse_args()

    if args.all_channels:
        print(f"Processando todos os canais de {args.edf}...")
        saved_paths = process_all_channels(
            args.edf, out_dir=args.out_dir, start_sec=args.start_sec,
            duration_sec=args.duration_sec, window_sec=args.window_sec,
            step_sec=args.step_sec)
        print(f"\n{len(saved_paths)} plots salvos em {args.out_dir}/")
        return

    print(f"Carregando {args.edf} (canal {args.channel})...")
    signal, sfreq, channel_name, seizure_intervals = load_channel(args.edf, args.channel)
    print(f"sfreq={sfreq}Hz, duração total={len(signal)/sfreq:.1f}s, "
          f"crises anotadas={seizure_intervals}")

    start_idx, end_idx, local_seizures = resolve_analysis_window(
        len(signal), sfreq, seizure_intervals, args.start_sec, args.duration_sec)
    segment = signal[start_idx:end_idx]

    print(f"Analisando {args.duration_sec:.0f}s a partir de t={start_idx/sfreq:.0f}s "
          f"com janelas de {args.window_sec}s (passo {args.step_sec}s)...")
    centers, lle_values = sliding_lle(
        segment, sfreq, window_sec=args.window_sec, step_sec=args.step_sec)
    print(f"{len(centers)} janelas calculadas. "
          f"λ₁ médio = {np.nanmean(lle_values):+.3f} nats/s")

    animate_lyapunov(segment, sfreq, centers, lle_values, local_seizures,
                      channel_name, save_path=args.save)


if __name__ == "__main__":
    main()
