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

from poincare_features import time_delay_embedding, SCRIPT_DIR, REPO_ROOT, DATASET_DIR

mne.set_log_level('ERROR')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Maior Expoente de Lyapunov (Rosenstein)
# ==========================================
def largest_lyapunov_exponent(signal, sfreq, d=5, tau=6, theiler_window=None,
                               trajectory_len=20, fit_range=None):
    """
    Estimates the Largest Lyapunov Exponent (LLE, lambda_1) of a signal
    using Rosenstein's (1993) method. The LLE measures how fast nearby
    trajectories in the reconstructed phase space diverge over time: a
    larger (more positive) LLE means more chaotic/divergent dynamics
    (typically background EEG), while a drop toward zero or negative values
    tends to coincide with more regular/synchronized dynamics (often seen
    at seizure onset) - which is why tracking it over a recording can help
    spot seizures.

    How, step by step:
      1. Reconstruct the phase space with `time_delay_embedding` - each
         point in `embedded` is one moment in the reconstructed dynamics.
      2. For every point i, find its nearest neighbor j in phase space,
         EXCLUDING points within `theiler_window` samples of i in time
         (the "Theiler window"). This exclusion matters: without it, the
         "nearest neighbor" of a point is almost always just the next
         sample in time (which is trivially close because the signal is
         continuous), not a genuinely different visit to a similar state -
         which would tell us nothing about chaotic divergence.
      3. For each valid pair (i, nearest neighbor j), track how their
         distance in phase space evolves over the next `trajectory_len`
         samples: d_k = ||embedded[i+k] - embedded[j+k]||. If the dynamics
         are chaotic, initially-close points should separate exponentially,
         i.e. log(d_k) should grow roughly linearly in k (for small k,
         before it saturates due to the attractor's finite size).
      4. Average log(d_k) over all valid pairs at each horizon k, then fit
         a straight line (`torch.linalg.lstsq`) to this average log-
         divergence curve vs. k (converted to seconds via `dt = 1/sfreq`).
         The slope of that line IS the estimated Lyapunov exponent.

    Args:
        signal: 1D array-like (numpy array or torch tensor) with one
            channel's raw samples for the segment to analyze.
        sfreq: sampling frequency of `signal`, in Hz (used to convert the
            trajectory-length horizon into seconds).
        d: embedding dimension for `time_delay_embedding`.
        tau: time lag (in samples) for `time_delay_embedding`.
        theiler_window: minimum time separation (in samples) required
            between a point and its "nearest neighbor" candidate. If None,
            defaults to max(tau * d, 5% of sfreq) - large enough to exclude
            temporally-adjacent (not dynamically meaningful) neighbors.
        trajectory_len: how many samples ahead to track the divergence
            between each pair - a fixed short horizon, since divergence
            only grows exponentially (and thus gives a meaningful slope)
            for a limited time before saturating.
        fit_range: optional (lo, hi) tuple restricting which horizons (as
            indices into the 0..trajectory_len range) are used for the
            final linear fit, in case the very earliest or latest horizons
            are noisy/saturated. If None, uses the full range.

    Returns:
        A Python float: the estimated Lyapunov exponent, in nats per second
        (natural log). Returns NaN if the segment is too short, has ~zero
        variance (flat signal), or doesn't have enough valid
        neighbor pairs / data points for a reliable estimate.
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
    Turns a single global LLE estimate into a time-varying curve, by
    computing `largest_lyapunov_exponent` independently on many short,
    overlapping windows slid across the full signal - so you can see how
    the LLE evolves over the course of a recording (e.g. dropping around a
    seizure) instead of getting just one number for the whole thing.

    Args:
        signal: 1D array-like with the full segment to analyze.
        sfreq: sampling frequency of `signal`, in Hz.
        window_sec: length of each LLE-estimation window, in seconds.
        step_sec: time step between consecutive window starts, in seconds
            (smaller than window_sec means overlapping windows, giving a
            smoother curve at the cost of more computation).
        d: embedding dimension, forwarded to `largest_lyapunov_exponent`.
        tau: time lag, forwarded to `largest_lyapunov_exponent`.

    Returns:
        Tuple (centers, values): both 1D numpy arrays of the same length.
        `centers` holds the time (in seconds, relative to the start of
        `signal`) of the midpoint of each window; `values` holds the
        estimated LLE for that window (NaN where the estimate was
        unreliable - see `largest_lyapunov_exponent`).
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
    """
    Looks up the annotated seizure intervals for one specific .edf recording
    from the global labels CSV, so plots can shade "this is when a seizure
    actually happened" for visual comparison against the LLE curve.

    Args:
        edf_path: path to the .edf file (only its basename is used to match
            against the CSV's 'file_name' column).
        labels_csv: path to the global labels CSV; defaults to
            `chb_mit_global_labels.csv` at the repo root if not given.

    Returns:
        List of (start_sec, end_sec) tuples, one per seizure interval
        annotated for this file. Empty list if the file has no seizures, or
        if `labels_csv` doesn't exist (so this function degrades gracefully
        rather than crashing when run without the labels CSV generated).
    """
    labels_csv = labels_csv or os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv')
    if not labels_csv or not os.path.exists(labels_csv):
        return []
    df = pd.read_csv(labels_csv)
    file_name = os.path.basename(edf_path)
    rows = df[(df['file_name'] == file_name) & (df['label'] == 1)]
    return list(zip(rows['start_sec'], rows['end_sec']))


def load_channel(edf_path, channel_name, labels_csv=None):
    """
    Loads a single channel's full signal from an .edf recording, along with
    its sampling frequency and annotated seizure intervals - the main entry
    point for getting real data into the rest of this script.

    Args:
        edf_path: path to the .edf file to load.
        channel_name: name of the channel to extract (e.g. 'P7-T7'). If not
            found exactly, falls back to the first channel whose name
            starts with `channel_name` (case-insensitively), to tolerate
            naming variants like "P7-T7-1".
        labels_csv: forwarded to `get_seizure_intervals`.

    Returns:
        Tuple (signal, sfreq, channel_name, seizure_intervals): `signal` is
        a 1D numpy array in microvolts (converted from the .edf's native
        Volts), `sfreq` is the sampling frequency in Hz, `channel_name` is
        the actual channel name used (may differ from the requested one if
        a fallback match was used), and `seizure_intervals` is the list
        from `get_seizure_intervals`.

    Raises:
        ValueError: if no channel matching `channel_name` (exactly or by
            prefix) exists in the recording.
    """
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
    Decides which [start_idx, end_idx) slice of a (possibly very long)
    recording to actually analyze/plot, and re-expresses the seizure
    intervals relative to that slice's own start (so seizure shading lines
    up correctly on a plot whose x-axis starts at 0, not at the original
    recording's timestamp).

    Args:
        full_len_samples: total length of the full recording, in samples.
        sfreq: sampling frequency, in Hz.
        seizure_intervals: list of (start_sec, end_sec) tuples in the full
            recording's original time reference (e.g. from
            `get_seizure_intervals`).
        start_sec: where to start the analysis window, in seconds (in the
            recording's original time reference). If None, defaults to 60
            seconds before the first annotated seizure, or 0 if there are no
            seizures.
        duration_sec: length of the analysis window, in seconds (clipped to
            the end of the recording if it would run past it).

    Returns:
        Tuple (start_idx, end_idx, local_seizures): `start_idx`/`end_idx`
        are sample indices into the full recording bounding the chosen
        window; `local_seizures` is a list of (start_sec, end_sec) tuples
        re-referenced so that 0 corresponds to `start_idx`, containing only
        the seizure intervals that actually overlap the chosen window.
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
    Builds the shared 2-panel figure layout (raw EEG on top, LLE curve on
    the bottom, sharing the same time x-axis) used by both the live
    animation (`animate_lyapunov`) and the static summary plot
    (`plot_final_summary`), so the two look consistent and the LLE panel
    always lines up in time with the EEG panel above it. Both panels get
    red shaded spans (`axvspan`) over any annotated seizure interval.

    Args:
        signal: 1D array-like with the full EEG segment being shown.
        sfreq: sampling frequency of `signal`, in Hz.
        seizure_intervals: list of (start_sec, end_sec) tuples, already
            re-referenced to this segment's own time axis (e.g. from
            `resolve_analysis_window`).
        channel_name: name of the channel being plotted, used in panel
            titles.
        title: overall figure title (`fig.suptitle`).

    Returns:
        Tuple (fig, ax_eeg, ax_lle, total_dur): the matplotlib Figure, the
        two Axes (EEG on top, LLE below, ready to have data plotted into
        them by the caller), and `total_dur` (the segment's duration in
        seconds, used to set the x-axis limits).
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
    Animates the LLE curve being "drawn" over time next to the full EEG
    trace, with a vertical cursor sweeping across both panels in sync - a
    way to see, moment by moment, how the estimated Lyapunov exponent
    behaves as the recording plays out (and whether it visibly dips around
    the shaded seizure interval).

    How: builds the shared 2-panel layout (`_build_two_panel_figure`), then
    uses `matplotlib.animation.FuncAnimation` to redraw, frame by frame: the
    LLE curve up to the current frame (`lle_line`), a vertical cursor at the
    current time on both panels, and a text readout of the current LLE
    value (turning red with a "CRISE" warning if the cursor is currently
    inside an annotated seizure interval).

    Args:
        signal: 1D array-like with the full EEG segment being animated.
        sfreq: sampling frequency of `signal`, in Hz.
        centers: 1D array of window-center timestamps, in seconds (from
            `sliding_lle`) - one animation frame per entry.
        lle_values: 1D array of LLE values matching `centers` (from
            `sliding_lle`).
        seizure_intervals: list of (start_sec, end_sec) tuples, re-
            referenced to this segment's own time axis.
        channel_name: name of the channel being animated, used in titles.
        save_path: if given, saves the animation as a .gif to this path
            (via `PillowWriter`) instead of displaying it interactively.

    Returns:
        None. Side effect: either shows the animation window
        (`plt.show()`) or writes a .gif file to `save_path`.
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
    Draws the same 2-panel layout as `animate_lyapunov` (EEG + LLE curve,
    shared time axis) but as a single static image with the FULL curve
    already drawn - much cheaper than animating, so this is what
    `process_all_channels` uses to produce one summary image per channel
    without the cost of rendering dozens of animations.

    Args:
        signal: 1D array-like with the full EEG segment being summarized.
        sfreq: sampling frequency of `signal`, in Hz.
        centers: 1D array of window-center timestamps, in seconds (from
            `sliding_lle`).
        lle_values: 1D array of LLE values matching `centers`.
        seizure_intervals: list of (start_sec, end_sec) tuples, re-
            referenced to this segment's own time axis.
        channel_name: name of the channel being plotted, used in titles.
        save_path: if given, saves the figure to this path (creating parent
            directories as needed) instead of displaying it interactively.

    Returns:
        None. Side effect: either shows the plot (`plt.show()`) or writes a
        PNG file to `save_path`.
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
    Batch-runs the sliding-window LLE analysis over EVERY channel in one
    .edf recording (rather than a single hand-picked channel), saving one
    static summary plot per channel - useful for scanning a whole recording
    to see which channels show the clearest LLE drop around a seizure,
    without manually re-running `main()` once per channel.

    Args:
        edf_path: path to the .edf file to process.
        out_dir: folder to save the per-channel PNGs into; defaults to
            `results/lyapunov_torch/` inside this script's folder.
        labels_csv: forwarded to `get_seizure_intervals`.
        start_sec: forwarded to `resolve_analysis_window` (same default:
            60s before the first seizure, or 0).
        duration_sec: forwarded to `resolve_analysis_window` - length of
            the analyzed segment, in seconds, same for every channel.
        window_sec: forwarded to `sliding_lle`.
        step_sec: forwarded to `sliding_lle`.

    Returns:
        List of the file paths (str) written, one PNG per channel, in the
        same order as the recording's channels.
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
    """
    Command-line entry point: parses CLI flags and either (a) runs
    `process_all_channels` over every channel of the given .edf file if
    `--all-channels` was passed, or (b) loads a single channel
    (`load_channel`), computes its sliding-window LLE (`sliding_lle`), and
    animates it (`animate_lyapunov`). See the `--help` output (printed from
    the argparse definitions below) for every flag's meaning and default.

    Takes no arguments (reads `sys.argv` via argparse) and returns nothing;
    all output is either printed to stdout, shown in a matplotlib window, or
    saved to disk depending on the flags passed.
    """
    parser = argparse.ArgumentParser(
        description="Estima e anima o Expoente de Lyapunov de um sinal de EEG (PyTorch).")
    parser.add_argument('--edf', default=os.path.join(DATASET_DIR, 'chb01', 'chb01_03.edf'),
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
