"""
Estima o Maior Expoente de Lyapunov (LLE) de um sinal de EEG usando o método
de Rosenstein (1993) em janelas deslizantes, e anima a evolução desse valor
lado a lado com a forma de onda bruta.

O LLE mede a taxa de divergência exponencial de trajetórias vizinhas no
espaço de fase reconstruído: valores mais altos indicam dinâmica mais
caótica (tipicamente o EEG de background), enquanto quedas do LLE tendem
a coincidir com o início de crises epilépticas (dinâmica mais sincronizada
e regular).
"""
import os
import argparse
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree

from poincare_features import time_delay_embedding

mne.set_log_level('ERROR')

# ==========================================
# 1. Maior Expoente de Lyapunov (Rosenstein)
# ==========================================
def largest_lyapunov_exponent(signal, sfreq, d=5, tau=6, theiler_window=None,
                               trajectory_len=20, fit_range=None):
    """
    Estima o maior expoente de Lyapunov de `signal` pelo método de Rosenstein.

    1. Reconstrói o espaço de fase por time-delay embedding.
    2. Para cada ponto, busca o vizinho mais próximo fora da janela de Theiler
       (evita pares que são apenas vizinhos temporais, não dinâmicos).
    3. Acompanha a divergência log(distância) entre cada par pelos próximos
       `trajectory_len` passos (uma contagem FIXA de amostras — não uma fração
       da janela — pois é o horizonte de curto prazo em que a divergência ainda
       cresce exponencialmente, antes de saturar por causa do tamanho finito
       do atrator).
    4. O expoente é a inclinação (ajuste linear) dessa curva de divergência média.

    Retorna o expoente em nats/segundo (log natural). np.nan se o segmento
    for curto/plano demais para uma estimativa confiável.
    """
    embedded = time_delay_embedding(signal, d=d, tau=tau)
    n = embedded.shape[0]
    if n < 20 or np.var(signal) <= 1e-12:
        return np.nan

    if theiler_window is None:
        theiler_window = max(tau * d, int(0.05 * sfreq))

    k_neighbors = min(n, theiler_window * 2 + 10)
    tree = cKDTree(embedded)
    _, idxs = tree.query(embedded, k=k_neighbors)

    # Para cada ponto i, primeiro vizinho fora da janela de Theiler
    nearest_idx = np.full(n, -1, dtype=int)
    offsets = np.abs(idxs - np.arange(n)[:, None])
    outside = offsets > theiler_window
    for i in range(n):
        candidates = np.where(outside[i])[0]
        if candidates.size:
            nearest_idx[i] = idxs[i, candidates[0]]

    valid = nearest_idx >= 0
    if valid.sum() < 10:
        return np.nan

    max_horizon = min(trajectory_len, n - 1)
    if max_horizon < 5:
        return np.nan

    log_div_sum = np.zeros(max_horizon)
    counts = np.zeros(max_horizon)

    for i in np.where(valid)[0]:
        j = nearest_idx[i]
        span = min(max_horizon, n - max(i, j))
        if span <= 0:
            continue
        diffs = embedded[i:i + span] - embedded[j:j + span]
        dist_k = np.linalg.norm(diffs, axis=1)
        dist_k[dist_k == 0] = 1e-10
        log_div_sum[:span] += np.log(dist_k)
        counts[:span] += 1

    has_data = counts > 0
    if has_data.sum() < 5:
        return np.nan

    mean_log_div = np.full(max_horizon, np.nan)
    mean_log_div[has_data] = log_div_sum[has_data] / counts[has_data]

    dt = 1.0 / sfreq  # o índice k do embedding avança uma amostra por passo
    lo, hi = fit_range if fit_range is not None else (0, max_horizon)
    hi = min(hi, max_horizon)
    fit_mask = has_data.copy()
    fit_mask[:lo] = False
    fit_mask[hi:] = False

    if fit_mask.sum() < 3:
        return np.nan

    k_axis = np.arange(max_horizon) * dt
    slope, _ = np.polyfit(k_axis[fit_mask], mean_log_div[fit_mask], 1)
    return slope


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
def load_channel(edf_path, channel_name, labels_csv='chb_mit_global_labels.csv'):
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

    seizure_intervals = []
    if labels_csv and os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        file_name = os.path.basename(edf_path)
        rows = df[(df['file_name'] == file_name) & (df['label'] == 1)]
        seizure_intervals = list(zip(rows['start_sec'], rows['end_sec']))

    return signal, sfreq, channel_name, seizure_intervals


# ==========================================
# 4. Animação: EEG + LLE evoluindo juntos
# ==========================================
def animate_lyapunov(signal, sfreq, centers, lle_values, seizure_intervals,
                      channel_name, eeg_window_sec=10.0, save_path=None):
    total_dur = len(signal) / sfreq

    fig, (ax_eeg, ax_lle) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(f"Expoente de Lyapunov ao vivo — canal {channel_name}",
                 fontsize=14, fontweight='bold')

    # --- Painel superior: EEG bruto, janela deslizante ---
    time_axis = np.arange(len(signal)) / sfreq
    eeg_line, = ax_eeg.plot([], [], color='darkblue', lw=1)
    cursor_eeg = ax_eeg.axvline(0, color='red', lw=1.5, alpha=0.8)
    ax_eeg.set_ylabel("Amplitude (uV)")
    ax_eeg.set_title("Sinal de EEG")
    ax_eeg.grid(True, alpha=0.3)
    for s0, s1 in seizure_intervals:
        ax_eeg.axvspan(s0, s1, color='red', alpha=0.15, label='Crise (anotada)')

    # --- Painel inferior: LLE crescendo ao longo do tempo ---
    lle_line, = ax_lle.plot([], [], color='teal', lw=1.8)
    cursor_lle = ax_lle.axvline(0, color='red', lw=1.5, alpha=0.8)
    ax_lle.set_xlim(0, total_dur)
    finite_vals = lle_values[np.isfinite(lle_values)]
    if finite_vals.size:
        pad = 0.1 * (finite_vals.max() - finite_vals.min() + 1e-6)
        ax_lle.set_ylim(finite_vals.min() - pad, finite_vals.max() + pad)
    ax_lle.set_xlabel("Tempo (s)")
    ax_lle.set_ylabel(r"$\lambda_1$ (nats/s)")
    ax_lle.set_title("Maior Expoente de Lyapunov (janela deslizante)")
    ax_lle.grid(True, alpha=0.3)
    for s0, s1 in seizure_intervals:
        ax_lle.axvspan(s0, s1, color='red', alpha=0.15)

    value_text = ax_lle.text(
        0.02, 0.92, "", transform=ax_lle.transAxes, fontsize=13,
        fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    def init():
        eeg_line.set_data([], [])
        lle_line.set_data([], [])
        value_text.set_text("")
        return eeg_line, lle_line, cursor_eeg, cursor_lle, value_text

    def update(frame_idx):
        t_now = centers[frame_idx]

        # Janela deslizante do EEG centrada no tempo atual
        lo_t = max(0.0, t_now - eeg_window_sec / 2.0)
        hi_t = lo_t + eeg_window_sec
        lo_i = int(lo_t * sfreq)
        hi_i = int(hi_t * sfreq)
        eeg_line.set_data(time_axis[lo_i:hi_i], signal[lo_i:hi_i])
        ax_eeg.set_xlim(lo_t, hi_t)
        finite_seg = signal[lo_i:hi_i]
        if finite_seg.size:
            margin = 0.1 * (np.ptp(finite_seg) + 1e-6)
            ax_eeg.set_ylim(finite_seg.min() - margin, finite_seg.max() + margin)

        # Curva de LLE crescendo até o frame atual
        lle_line.set_data(centers[:frame_idx + 1], lle_values[:frame_idx + 1])

        cursor_eeg.set_xdata([t_now, t_now])
        cursor_lle.set_xdata([t_now, t_now])

        current_val = lle_values[frame_idx]
        label = f"t = {t_now:5.1f}s   λ₁ = {current_val:+.3f} nats/s" \
            if np.isfinite(current_val) else f"t = {t_now:5.1f}s   λ₁ = N/D"
        in_seizure = any(s0 <= t_now <= s1 for s0, s1 in seizure_intervals)
        value_text.set_text(("⚠ CRISE — " if in_seizure else "") + label)
        value_text.set_color('red' if in_seizure else 'black')

        return eeg_line, lle_line, cursor_eeg, cursor_lle, value_text

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


# ==========================================
# 5. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Estima e anima o Expoente de Lyapunov de um sinal de EEG.")
    parser.add_argument('--edf', default='dataset_chbmit/chb01/chb01_03.edf',
                         help="Caminho do arquivo .edf")
    parser.add_argument('--channel', default='P7-T7', help="Nome do canal a analisar")
    parser.add_argument('--start-sec', type=float, default=None,
                         help="Início do trecho analisado (s). Padrão: 60s antes da crise, ou 0.")
    parser.add_argument('--duration-sec', type=float, default=180.0,
                         help="Duração do trecho analisado, em segundos.")
    parser.add_argument('--window-sec', type=float, default=2.0,
                         help="Tamanho da janela usada em cada estimativa de LLE.")
    parser.add_argument('--step-sec', type=float, default=0.5,
                         help="Passo entre janelas consecutivas.")
    parser.add_argument('--eeg-window-sec', type=float, default=10.0,
                         help="Largura da janela de EEG mostrada na animação.")
    parser.add_argument('--save', default=None,
                         help="Se definido, salva a animação como .gif neste caminho em vez de exibi-la.")
    args = parser.parse_args()

    print(f"Carregando {args.edf} (canal {args.channel})...")
    signal, sfreq, channel_name, seizure_intervals = load_channel(args.edf, args.channel)
    print(f"sfreq={sfreq}Hz, duração total={len(signal)/sfreq:.1f}s, "
          f"crises anotadas={seizure_intervals}")

    start_sec = args.start_sec
    if start_sec is None:
        start_sec = max(0.0, seizure_intervals[0][0] - 60) if seizure_intervals else 0.0
    start_idx = int(start_sec * sfreq)
    end_idx = min(len(signal), start_idx + int(args.duration_sec * sfreq))
    segment = signal[start_idx:end_idx]

    # Re-referencia os intervalos de crise para o começo do trecho analisado
    local_seizures = [
        (max(0.0, s0 - start_sec), s1 - start_sec)
        for s0, s1 in seizure_intervals
        if s1 > start_sec and s0 < start_sec + args.duration_sec
    ]

    print(f"Analisando {args.duration_sec:.0f}s a partir de t={start_sec:.0f}s "
          f"com janelas de {args.window_sec}s (passo {args.step_sec}s)...")
    centers, lle_values = sliding_lle(
        segment, sfreq, window_sec=args.window_sec, step_sec=args.step_sec)
    print(f"{len(centers)} janelas calculadas. "
          f"λ₁ médio = {np.nanmean(lle_values):+.3f} nats/s")

    animate_lyapunov(segment, sfreq, centers, lle_values, local_seizures,
                      channel_name, eeg_window_sec=args.eeg_window_sec,
                      save_path=args.save)


if __name__ == "__main__":
    main()
