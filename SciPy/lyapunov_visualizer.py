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
import time
import argparse
import numpy as np
import nolds
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from poincare_features import time_delay_embedding

mne.set_log_level('ERROR')

# ==========================================
# 1. Maior Expoente de Lyapunov (Rosenstein)
# ==========================================
def mean_period(signal):
    """
    Período médio do sinal, em amostras, estimado pela FFT (Rosenstein, 1993).

    A frequência média é a média das frequências do espectro ponderada pela
    potência de cada uma (o componente DC, frequência 0, é ignorado). O
    período médio é o inverso dela: quantas amostras dura, em média, um
    "ciclo" do sinal.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal))    # em ciclos por amostra
    power = np.abs(spectrum) ** 2
    mean_freq = np.sum(freqs[1:] * power[1:]) / np.sum(power[1:])
    period = int(np.ceil(1.0 / mean_freq))
    
    return period


def largest_lyapunov_exponent(signal, sfreq, d=5, tau=6,
                               trajectory_len=20):
    """
    Estima o maior expoente de Lyapunov de `signal` pelo método de Rosenstein.

    1. Estima o período médio do sinal pela FFT (`mean_period`).
    2. Reconstrói o espaço de fase por time-delay embedding.
    3. Para cada ponto, busca o vizinho mais próximo (distância euclidiana)
       entre os pontos separados dele no tempo por MAIS que o período médio
       — assim o vizinho vem de outro trecho do sinal, e não é só a amostra
       seguinte da mesma trajetória.
    4. Acompanha a divergência log(distância) entre cada par pelos próximos
       `trajectory_len` passos (uma contagem FIXA de amostras — não uma fração
       da janela — pois é o horizonte de curto prazo em que a divergência ainda
       cresce exponencialmente, antes de saturar por causa do tamanho finito
       do atrator).
    5. O expoente é a inclinação (ajuste linear) dessa curva de divergência média.

    Retorna o expoente em nats/segundo (log natural). np.nan se o segmento
    for curto/plano demais para uma estimativa confiável.
    """
    embedded = time_delay_embedding(signal, d=d, tau=tau)
    n = embedded.shape[0]
    if n < 20 or np.var(signal) <= 1e-12:
        return np.nan

    # Matriz de distâncias euclidianas entre todos os pares de pontos (n x n)
    diffs_all = embedded[:, None, :] - embedded[None, :, :]
    dist_matrix = np.sqrt(np.sum(diffs_all ** 2, axis=-1))

    # Restrição de separação temporal: pontos a até `period` amostras de
    # distância no tempo não podem ser vizinhos (inclui o próprio ponto, |i-j|=0)
    period = mean_period(signal)
    idx = np.arange(n)
    too_close_in_time = np.abs(idx[:, None] - idx[None, :]) <= period
    dist_matrix[too_close_in_time] = np.inf
    nearest_idx = np.argmin(dist_matrix, axis=1)

    max_horizon = min(trajectory_len, n - 1)
    if max_horizon < 5:
        return np.nan

    log_div_sum = np.zeros(max_horizon)
    counts = np.zeros(max_horizon)

    for i in range(n):
        j = nearest_idx[i]
        span = min(max_horizon, n - max(i, j))
        if span <= 0:
            continue
        diffs = embedded[i:i + span] - embedded[j:j + span]
        dist_k = np.linalg.norm(diffs, axis=1)
        dist_k[dist_k == 0] = 1e-10
        log_div_sum[:span] += np.log(dist_k)
        counts[:span] += 1

    # Passos k em que pelo menos um par contribuiu (evita divisão 0/0).
    # Como todo par contribui a partir de k=0, esses passos são sempre
    # os primeiros: k = 0, 1, 2, ...
    has_data = counts > 0
    if has_data.sum() < 5:
        return np.nan

    # Curva de divergência: média de ln(distância) após k passos
    mean_log_div = log_div_sum[has_data] / counts[has_data]

    # Eixo X em segundos: o passo k corresponde ao tempo k / sfreq
    time_axis = np.arange(len(mean_log_div)) / sfreq

    # λ = inclinação da reta ajustada por mínimos quadrados
    slope, _ = np.polyfit(time_axis, mean_log_div, 1)
    return slope


def largest_lyapunov_exponent_nolds(signal, sfreq, d=5, tau=6,
                                     trajectory_len=20, fit='poly'):
    """
    Mesma estimativa de `largest_lyapunov_exponent`, mas usando a implementação
    de referência `nolds.lyap_r` (também baseada em Rosenstein, 1993), para
    validar a implementação própria.

    Os parâmetros são mapeados para ficarem equivalentes aos da versão própria:
      - emb_dim   <- d               (dimensão de embedding)
      - lag       <- tau             (atraso do embedding, em amostras)
      - min_tsep  <- None            (o nolds calcula a janela de Theiler
                                      automaticamente como o período médio do
                                      sinal = 1 / frequência média do espectro)
      - tau       <- 1/sfreq         (passo de tempo entre amostras -> resultado em nats/s)
      - fit='poly' (mínimos quadrados, como o np.polyfit da versão própria;
        o padrão do nolds é 'RANSAC', que é robusto a outliers mas exige sklearn)

    Retorna o expoente em nats/segundo, ou np.nan se o nolds não conseguir
    estimar (segmento curto/plano demais, vizinhos insuficientes, etc.).
    """
    if len(signal) < 20 or np.var(signal) <= 1e-12:
        return np.nan

    try:
        return nolds.lyap_r(
            np.asarray(signal, dtype=float), emb_dim=d, lag=tau,
            min_tsep=None, tau=1.0 / sfreq,
            trajectory_len=trajectory_len, fit=fit)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


# ==========================================
# 2. LLE em janelas deslizantes sobre a gravação
# ==========================================
def sliding_lle(signal, sfreq, window_sec=2.0, step_sec=0.5, d=5, tau=6,
                estimator=largest_lyapunov_exponent):
    """
    Aplica `estimator` (por padrão `largest_lyapunov_exponent`; também aceita
    `largest_lyapunov_exponent_nolds`) em janelas deslizantes ao longo do sinal.
    Retorna (centros_em_segundos, valores_do_lle).
    """
    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(step_sec * sfreq))

    centers, values = [], []
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        segment = signal[start:start + window_samples]
        lle = estimator(segment, sfreq, d=d, tau=tau)
        centers.append((start + window_samples / 2.0) / sfreq)
        values.append(lle)

    return np.array(centers), np.array(values)


# ==========================================
# 3. Carregamento de dados (formato CHB-MIT)
# ==========================================
def get_seizure_intervals(edf_path, labels_csv='chb_mit_global_labels.csv'):
    if not labels_csv or not os.path.exists(labels_csv):
        return []
    df = pd.read_csv(labels_csv)
    file_name = os.path.basename(edf_path)
    rows = df[(df['file_name'] == file_name) & (df['label'] == 1)]
    return list(zip(rows['start_sec'], rows['end_sec']))


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
    o EEG completo em cima e o espaço para a curva de LLE embaixo. Usada tanto
    pela animação ao vivo quanto pelo plot estático final, para que as duas
    visualizações fiquem sempre na mesma escala de tempo.
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
    escala de tempo (eixo X compartilhado) — um cursor vertical percorre os
    dois painéis simultaneamente. Ao final, o quadro exibido já é o resumo
    completo: sinal inteiro vs. evolução inteira do LLE.
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

        return lle_line, cursor_eeg, cursor_lle, value_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(centers), init_func=init,
        interval=80, blit=False, repeat=False
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        writer = animation.PillowWriter(fps=12)
        ani.save(save_path, writer=writer)
        print(f"Animação salva em: {save_path}")
    else:
        plt.show()


def plot_final_summary(signal, sfreq, centers, lle_values, seizure_intervals,
                        channel_name, save_path=None):
    """
    Plot estático (sem animação) do sinal completo vs. a evolução completa do
    LLE, na mesma escala de tempo. Usado para salvar rapidamente um resumo por
    canal (animar dezenas de canais seria lento e desnecessário).
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
def process_all_channels(edf_path, out_dir='results/lyapunov',
                          labels_csv='chb_mit_global_labels.csv',
                          start_sec=None, duration_sec=180.0,
                          window_sec=2.0, step_sec=0.5):
    """
    Roda a estimativa de LLE em janela deslizante para TODOS os canais do
    arquivo .edf (não apenas um) e salva, para cada canal, o plot final
    estático (sinal completo vs. evolução completa do LLE) em `out_dir`.
    """
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
# 6. Comparação: implementação própria vs. nolds.lyap_r
# ==========================================
def compare_with_nolds(signal, sfreq, seizure_intervals, channel_name,
                        window_sec=2.0, step_sec=0.5, save_path=None):
    """
    Calcula o LLE em janelas deslizantes com as duas implementações
    (`largest_lyapunov_exponent` e `largest_lyapunov_exponent_nolds`), imprime
    métricas de concordância (média, correlação de Pearson, erro absoluto médio,
    tempo de execução) e plota as duas curvas sobrepostas, abaixo do EEG.
    """
    results = {}
    for name, estimator in [('Própria', largest_lyapunov_exponent),
                            ('nolds.lyap_r', largest_lyapunov_exponent_nolds)]:
        t0 = time.perf_counter()
        centers, values = sliding_lle(signal, sfreq, window_sec=window_sec,
                                      step_sec=step_sec, estimator=estimator)
        results[name] = (centers, values, time.perf_counter() - t0)

    centers, own_vals, own_time = results['Própria']
    _, nolds_vals, nolds_time = results['nolds.lyap_r']

    both = np.isfinite(own_vals) & np.isfinite(nolds_vals)
    if both.sum() >= 2:
        pearson_r = np.corrcoef(own_vals[both], nolds_vals[both])[0, 1]
        mae = np.mean(np.abs(own_vals[both] - nolds_vals[both]))
    else:
        pearson_r, mae = np.nan, np.nan

    print("\n=== Comparação: implementação própria vs. nolds.lyap_r ===")
    print(f"Janelas válidas em ambas: {both.sum()}/{len(centers)}")
    print(f"λ₁ médio (própria) = {np.nanmean(own_vals):+.3f} nats/s  "
          f"[{own_time:.1f}s de execução]")
    print(f"λ₁ médio (nolds)   = {np.nanmean(nolds_vals):+.3f} nats/s  "
          f"[{nolds_time:.1f}s de execução]")
    print(f"Correlação de Pearson = {pearson_r:.3f}")
    print(f"Erro absoluto médio   = {mae:.3f} nats/s")

    fig, ax_eeg, ax_lle, _ = _build_two_panel_figure(
        signal, sfreq, seizure_intervals, channel_name,
        f"LLE: implementação própria vs. nolds.lyap_r — canal {channel_name}")
    ax_lle.plot(centers, own_vals, color='teal', lw=1.5, label='Própria (Rosenstein)')
    ax_lle.plot(centers, nolds_vals, color='darkorange', lw=1.5, ls='--',
                label='nolds.lyap_r')
    ax_lle.legend(loc='upper right')
    ax_lle.text(
        0.02, 0.92, f"Pearson r = {pearson_r:.3f}   MAE = {mae:.3f} nats/s",
        transform=ax_lle.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Plot salvo em: {save_path}")
    else:
        plt.show()

    return centers, own_vals, nolds_vals


# ==========================================
# 7. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Estima e anima o Expoente de Lyapunov de um sinal de EEG.")
    parser.add_argument('--edf', default='dataset_chbmit/chb01/chb01_03.edf',
                         help="Caminho do arquivo .edf")
    parser.add_argument('--channel', default='P7-T7',
                         help="Nome do canal a analisar (ignorado com --all-channels).")
    parser.add_argument('--all-channels', action='store_true',
                         help="Processa TODOS os canais do arquivo e salva um plot final "
                              "por canal, em vez de animar um único canal.")
    parser.add_argument('--out-dir', default='results/lyapunov',
                         help="Pasta onde salvar os plots quando --all-channels for usado.")
    parser.add_argument('--start-sec', type=float, default=None,
                         help="Início do trecho analisado (s). Padrão: 60s antes da crise, ou 0.")
    parser.add_argument('--duration-sec', type=float, default=180.0,
                         help="Duração do trecho analisado, em segundos.")
    parser.add_argument('--window-sec', type=float, default=2.0,
                         help="Tamanho da janela usada em cada estimativa de LLE.")
    parser.add_argument('--step-sec', type=float, default=0.5,
                         help="Passo entre janelas consecutivas.")
    parser.add_argument('--save', nargs='?', const='lyapunov_animation.gif', default=None,
                    help="Salva a animação como .gif no caminho dado. "
                         "Sem valor, usa 'lyapunov_animation.gif'. "
                         "Omitido, exibe a animação na tela.")
    parser.add_argument('--compare-nolds', nargs='?', const='', default=None,
                         metavar='PNG',
                         help="Compara a implementação própria com nolds.lyap_r no canal "
                              "escolhido. Com um caminho, salva o plot como .png; sem "
                              "valor, exibe na tela.")
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

    if args.compare_nolds is not None:
        compare_with_nolds(segment, sfreq, local_seizures, channel_name,
                           window_sec=args.window_sec, step_sec=args.step_sec,
                           save_path=args.compare_nolds or None)
        return

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
