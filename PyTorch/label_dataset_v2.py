import mne
import numpy as np
import pandas as pd
import os
import torch
from joblib import Parallel, delayed

mne.set_log_level('ERROR')  # This forces MNE to only print fatal errors, hiding the warnings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Canais padrão do sistema 10-20 (Montagem comum no CHB-MIT)
CHANNELS_TO_KEEP = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def torch_skew(x):
    x = x - x.mean()
    m2 = (x ** 2).mean()
    m3 = (x ** 3).mean()
    return m3 / (m2 ** 1.5 + 1e-12)


def torch_kurtosis(x):
    x = x - x.mean()
    m2 = (x ** 2).mean()
    m4 = (x ** 4).mean()
    return m4 / (m2 ** 2 + 1e-12) - 3.0


def welch_psd_torch(signal, sfreq, nperseg):
    """
    Densidade espectral de potência via método de Welch, reimplementada com
    torch.stft (substitui scipy.signal.welch): janela Hann, 50% de overlap,
    remoção de média por segmento ('constant' detrend) e escala 'density',
    igual aos defaults do scipy.
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
    df = pd.read_csv('chb_mit_global_labels.csv')

    base_path = './dataset_chbmit'
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
            torch.save(X_stacked, f'X_{patient_id}.pt')
            torch.save(y_stacked, f'y_{patient_id}.pt')
            print(f"Salvo {patient_id}: X={tuple(X_stacked.shape)}")
