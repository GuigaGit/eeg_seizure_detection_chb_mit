import mne
import numpy as np
import pandas as pd
import os
import torch
from joblib import Parallel, delayed

mne.set_log_level('ERROR')  # This forces MNE to only print fatal errors, hiding the warnings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caminhos resolvidos a partir da localização do script (não do cwd), para
# que main_pipeline.py funcione tanto rodando de dentro de PyTorch/ quanto
# da raiz do repositório.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Pasta com os arquivos .edf do CHB-MIT. Cada máquina pode ter o dataset em
# um lugar diferente, então isso é configurável via variável de ambiente:
#   export CHB_MIT_DATASET_DIR=/caminho/para/dataset_chbmit
# Se não for definida, usa a pasta de exemplo dentro do próprio repositório.
DATASET_DIR = os.environ.get('CHB_MIT_DATASET_DIR', os.path.join(REPO_ROOT, 'dataset_chbmit'))

# Canais padrão do sistema 10-20 (Montagem comum no CHB-MIT)
CHANNELS_TO_KEEP = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def torch_skew(x):
    """
    Computes the (biased/population) skewness of a signal - a measure of how
    asymmetric its amplitude distribution is around the mean. Positive means
    a longer tail toward high values, negative toward low values, ~0 means
    roughly symmetric (torch equivalent of scipy.stats.skew with its default
    bias=True, i.e. no small-sample correction).

    How: standardizes the signal (subtracts the mean) and computes the
    third standardized moment: mean(x^3) / mean(x^2)^1.5. A tiny epsilon is
    added to the denominator to avoid a divide-by-zero on a perfectly flat
    signal.

    Args:
        x: 1D torch.Tensor with one channel's samples (already float).

    Returns:
        0-dimensional torch.Tensor with the skewness value.
    """
    x = x - x.mean()
    m2 = (x ** 2).mean()
    m3 = (x ** 3).mean()
    return m3 / (m2 ** 1.5 + 1e-12)


def torch_kurtosis(x):
    """
    Computes the (biased, excess) kurtosis of a signal - how heavy-tailed /
    peaked its amplitude distribution is compared to a Gaussian. 0 means
    "as Gaussian as it gets", positive means more extreme outliers than a
    Gaussian, negative means a flatter/more uniform-looking distribution
    (torch equivalent of scipy.stats.kurtosis with its defaults: bias=True,
    fisher=True i.e. "excess" kurtosis with the Gaussian's 3.0 subtracted
    off).

    How: standardizes the signal and computes the fourth standardized
    moment, mean(x^4) / mean(x^2)^2, then subtracts 3 (the kurtosis of a
    Gaussian) so that 0 is the "no excess" baseline.

    Args:
        x: 1D torch.Tensor with one channel's samples (already float).

    Returns:
        0-dimensional torch.Tensor with the excess kurtosis value.
    """
    x = x - x.mean()
    m2 = (x ** 2).mean()
    m4 = (x ** 4).mean()
    return m4 / (m2 ** 2 + 1e-12) - 3.0


def welch_psd_torch(signal, sfreq, nperseg):
    """
    Estimates the Power Spectral Density (PSD) of a signal using Welch's
    method (torch replacement for scipy.signal.welch, matching its default
    settings: Hann window, 50% overlap, per-segment mean removal
    ('constant' detrend), and 'density' scaling).

    Why Welch's method: a single FFT of a noisy signal gives a very noisy
    spectrum estimate. Welch's method instead splits the signal into
    overlapping segments, computes a periodogram (squared FFT magnitude) for
    each, and averages them - trading frequency resolution for a much more
    stable (lower-variance) power estimate.

    How, step by step:
      1. Split `signal` into overlapping windows of length `nperseg`
         (50% overlap, i.e. step = nperseg // 2) using `Tensor.unfold`.
      2. Remove each segment's own mean (detrend='constant' in scipy) so a
         DC offset doesn't leak spectral power into low frequencies.
      3. Multiply each segment by a Hann window to reduce spectral leakage
         from the sharp edges of a finite segment.
      4. Take the real FFT of every windowed segment and square its
         magnitude to get one periodogram per segment.
      5. Apply the 'density' scaling factor (1 / (fs * sum(window^2))) and
         double all frequencies except DC and Nyquist, since the real FFT
         only keeps the non-negative half of the spectrum but the power of
         a real signal is split between positive and negative frequencies.
      6. Average the periodograms across segments to get the final PSD.

    Args:
        signal: 1D torch.Tensor with one channel's samples for the window
            being analyzed.
        sfreq: sampling frequency of `signal`, in Hz.
        nperseg: length of each Welch segment, in samples (the reference
            pipeline uses nperseg == sfreq, i.e. 1-second segments).

    Returns:
        Tuple (freqs, psd_mean): `freqs` is a 1D torch.Tensor of the
        frequency bins (Hz), and `psd_mean` is a 1D torch.Tensor of the same
        length with the estimated power at each frequency (units: signal^2
        / Hz, matching scipy's 'density' scaling).
    """
    n = signal.shape[0]
    if n < nperseg:
        nperseg = n
    noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)

    segments = signal.unfold(0, nperseg, step)
    segments = segments - segments.mean(dim=1, keepdim=True)

    window = torch.hann_window(nperseg, periodic=False, device=signal.device)
    windowed = segments * window

    fft = torch.fft.rfft(windowed, dim=1)
    psd = fft.abs() ** 2

    scale = 1.0 / (sfreq * (window ** 2).sum())
    psd = psd * scale
    if nperseg % 2 == 0:
        psd[:, 1:-1] *= 2
    else:
        psd[:, 1:] *= 2

    psd_mean = psd.mean(dim=0)
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sfreq).to(signal.device)
    return freqs, psd_mean


def extract_all_features(window_data, sfreq):
    """
    Extracts classic time- and frequency-domain features from every channel
    of an EEG window, and concatenates them into one flat feature vector
    (the "classic features" alternative to the Poincaré-based features in
    poincare_features.py).

    Per channel, computes 6 time-domain features:
      - mean: DC level of the window.
      - variance: overall signal power/spread.
      - skewness (`torch_skew`): amplitude-distribution asymmetry.
      - kurtosis (`torch_kurtosis`): amplitude-distribution peakedness.
      - RMS (root-mean-square): overall amplitude scale.
      - sum(|diff|): total absolute sample-to-sample change - despite the
        variable naming in earlier drafts suggesting "zero-crossings", this
        is actually a total-variation measure, not a zero-crossing count.
    ...followed by 5 frequency-domain features: the power in each classic
    EEG band (delta 0.5-4Hz, theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz,
    gamma 30-45Hz), computed by integrating the Welch PSD (`welch_psd_torch`)
    over each band's frequency range with the trapezoidal rule.

    Args:
        window_data: array-like of shape (n_channels, n_samples) - one
            short EEG window with all channels for that window.
        sfreq: sampling frequency of the window, in Hz.

    Returns:
        1D torch.Tensor of length 11 * n_channels (6 time-domain + 5
        frequency-domain features per channel), channels concatenated in
        their input order.
    """
    all_features = []
    nperseg = sfreq
    for channel_signal in window_data:
        x = torch.as_tensor(channel_signal, dtype=torch.float32, device=DEVICE)

        # Domínio do Tempo
        all_features.append(x.mean())
        all_features.append(x.var(unbiased=False))
        all_features.append(torch_skew(x))
        all_features.append(torch_kurtosis(x))
        all_features.append(torch.sqrt((x ** 2).mean()))
        all_features.append(torch.diff(x).abs().sum())

        # Domínio da Frequência (Bandas de EEG)
        freqs, psd = welch_psd_torch(x, sfreq, nperseg)
        for fmin, fmax in BANDS:
            mask = (freqs >= fmin) & (freqs <= fmax)
            all_features.append(torch.trapezoid(psd[mask], freqs[mask]))

    return torch.stack(all_features)


def process_single_file(file_name, group, base_path, window_sec, sfreq):
    """
    Builds the labeled classic-feature dataset for a single .edf recording:
    loads it, extracts seizure windows and a matching sample of background
    windows, and turns each window into a feature vector via
    `extract_all_features`. (Same overall structure as
    `poincare_features.process_single_file`, but requires only 15 of the 18
    target channels to be present instead of all of them, and uses
    `extract_all_features` instead of the Poincaré features.)

    Args:
        file_name: name of the .edf file to process (e.g. "chb01_03.edf").
        group: pandas DataFrame slice (rows of the global labels CSV for
            this file) with columns 'patient', 'label', 'start_sec',
            'end_sec' describing the seizure intervals annotated in it.
        base_path: root folder containing one subfolder per patient
            (typically `DATASET_DIR`), used to locate the .edf file.
        window_sec: length of each feature-extraction window, in seconds.
        sfreq: sampling frequency of the recording, in Hz.

    Returns:
        Tuple (X_local, y_local): X_local is a torch.Tensor of shape
        (n_windows, 11 * n_channels) with one feature vector per window,
        y_local is a torch.Tensor of shape (n_windows,) with the matching
        0/1 labels. Returns None if the file is missing, has fewer than 15
        of the required channels, or fails to load/process.
    """
    X_local, y_local = [], []
    patient = group['patient'].iloc[0]
    path_edf = os.path.join(base_path, patient, file_name)

    if not os.path.exists(path_edf):
        return None

    try:
        # 1. Carrega o arquivo permitindo nomes duplicados (ele vai renomear automaticamente)
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)

        # 2. Limpeza de nomes: remove espaços e converte para maiúsculas para facilitar a comparação
        raw.rename_channels(lambda x: x.strip().upper())

        # 3. Mapeamento de sinonimos comuns no CHB-MIT (Ex: T8-P8 as vezes vem com -1 ou .1)
        existing_channels = raw.ch_names
        final_selection = []

        target_channels = [c.upper() for c in CHANNELS_TO_KEEP]

        for target in target_channels:
            if target in existing_channels:
                final_selection.append(target)
            else:
                found_alt = [ch for ch in existing_channels if ch.startswith(target)]
                if found_alt:
                    final_selection.append(found_alt[0])

        # 4. Verifica se temos canais suficientes para treinar (ex: pelo menos 15 dos 18)
        if len(final_selection) < 15:
            print(f"Erro em {file_name}: Apenas {len(final_selection)} canais encontrados. Pulando.")
            return None

        # 5. Usa o novo método .pick (o pick_channels é legado)
        raw.pick(final_selection)

        # 6. Garante que agora todos tenham EXATAMENTE os mesmos nomes
        rename_dict = {actual: original for actual, original in zip(raw.ch_names, CHANNELS_TO_KEEP)}
        raw.rename_channels(rename_dict)

        data = raw.get_data()

        janela_samples = int(window_sec * sfreq)
        seizures = group[group['label'] == 1]

        is_seizure = np.zeros(data.shape[1], dtype=bool)

        # Extract Seizure Windows (Label 1)
        for _, row in seizures.iterrows():
            start_idx = int(row['start_sec'] * sfreq)
            end_idx = int(row['end_sec'] * sfreq)

            is_seizure[start_idx:end_idx] = True

            for start in range(start_idx, end_idx - janela_samples, janela_samples):
                window = data[:, start:start + janela_samples]
                X_local.append(extract_all_features(window, sfreq))
                y_local.append(1)

        # Background (Label 0)
        cont_bg = 0
        while cont_bg < 15:
            start = np.random.randint(0, data.shape[1] - janela_samples)
            if not np.any(is_seizure[start: start + janela_samples]):
                window = data[:, start:start + janela_samples]
                X_local.append(extract_all_features(window, sfreq))
                y_local.append(0)
                cont_bg += 1

        return torch.stack(X_local), torch.tensor(y_local, dtype=torch.long)

    except Exception as e:
        print(f"Erro em {file_name}: {e}")
        return None


def build_complete_dataset(base_path, global_labels_csv, window_sec=4):
    """
    Builds one combined dataset (all patients, all files together) of
    classic features, by running `process_single_file` in parallel over
    every file listed in the global labels CSV and concatenating the
    results. Unlike the `__main__` block below (which saves one .pt pair
    per patient), this returns everything as a single pair of tensors -
    useful for training a single global (not per-patient) model.

    Args:
        base_path: root folder containing one subfolder per patient.
        global_labels_csv: path to the CSV with columns 'file_name',
            'patient', 'label', 'start_sec', 'end_sec' (as produced by
            global_parse_dataset.py).
        window_sec: length of each feature-extraction window, in seconds.

    Returns:
        Tuple (X, y): X is a torch.Tensor of shape
        (total_n_windows, 11 * n_channels) stacking every window from every
        file, y is a torch.Tensor of shape (total_n_windows,) with the
        matching 0/1 labels.
    """
    df = pd.read_csv(global_labels_csv)
    sfreq = 256

    results = Parallel(n_jobs=-1)(
        delayed(process_single_file)(f, g, base_path, window_sec, sfreq)
        for f, g in df.groupby('file_name')
    )

    X, y = [], []
    for res in results:
        if res is not None:
            X.append(res[0])
            y.append(res[1])

    return torch.cat(X, dim=0), torch.cat(y, dim=0)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv'))

    base_path = DATASET_DIR
    window_sec = 4
    sfreq = 256

    # Process per patient instead of globally
    for patient_id, group in df.groupby('patient'):
        print(f"Processando {patient_id}...")

        results = Parallel(n_jobs=-1)(
            delayed(process_single_file)(f, file_group, base_path, window_sec, sfreq)
            for f, file_group in group.groupby('file_name')
        )

        X_patient, y_patient = [], []
        for res in results:
            if res is not None:
                X_patient.append(res[0])
                y_patient.append(res[1])

        if X_patient:
            X_stacked = torch.cat(X_patient, dim=0)
            y_stacked = torch.cat(y_patient, dim=0)
            torch.save(X_stacked, os.path.join(SCRIPT_DIR, f'X_{patient_id}.pt'))
            torch.save(y_stacked, os.path.join(SCRIPT_DIR, f'y_{patient_id}.pt'))
            print(f"Salvo {patient_id}: X={tuple(X_stacked.shape)}")
