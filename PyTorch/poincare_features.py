import mne
import numpy as np
import pandas as pd
import os
import torch
from joblib import Parallel, delayed
import warnings

# This forces MNE to only print fatal errors, hiding the warnings
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Canais padrão do sistema 10-20 (Montagem comum no CHB-MIT - 23 canais)
CHANNELS_TO_KEEP = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
]

# ==========================================
# 1. Phase Space Reconstruction (PSR)
# ==========================================
def time_delay_embedding(signal, d=5, tau=6):
    """
    Reconstructs the phase space using time-delay embedding (PyTorch version).
    """
    signal = torch.as_tensor(signal, dtype=torch.float32, device=DEVICE)
    n_samples = signal.shape[0]
    valid_length = n_samples - (d - 1) * tau

    if valid_length <= 0:
        # Fallback to prevent crashes on extremely short signal edges
        return torch.zeros((1, d), device=DEVICE)

    embedded = torch.stack(
        [signal[i * tau: valid_length + i * tau] for i in range(d)], dim=1
    )
    return embedded

# ==========================================
# 2. PCA & Poincaré Section Mapping
# ==========================================
def pca_transform(embedded, n_components=None):
    """
    PCA via SVD (replaces sklearn.decomposition.PCA), sorted by descending
    explained variance, same convention as sklearn's fit_transform.
    """
    mean = embedded.mean(dim=0, keepdim=True)
    centered = embedded - mean
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    if n_components is not None:
        Vh = Vh[:n_components]
    return centered @ Vh.T


def fit_line(x, y):
    """Least-squares fit of y = m*x + c (torch equivalent of np.polyfit(x, y, 1))."""
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    return solution[0, 0], solution[1, 0]


def get_poincare_intersections(embedded_space):
    """
    Applies PCA, fits a 1st-degree polynomial (line), and finds intersections.
    Returns the PC1 values of the intersection points.
    """
    if embedded_space.shape[0] < 2:
        return torch.empty(0, device=DEVICE)

    # Check variance BEFORE PCA to avoid divide-by-zero on flat signals
    if embedded_space.var() <= 1e-12:
        return torch.empty(0, device=DEVICE)

    pcs = pca_transform(embedded_space, n_components=2)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]

    if pc1.var() == 0:
        return torch.empty(0, device=DEVICE)

    m, c = fit_line(pc1, pc2)

    # Vectorized sign-crossing search (the sequential Python loop from the
    # SciPy version becomes a single set of tensor ops here).
    f = pc2 - (m * pc1 + c)
    crossings = torch.nonzero((f[:-1] * f[1:]) < 0, as_tuple=True)[0]
    if crossings.numel() == 0:
        return torch.empty(0, device=DEVICE)

    f1, f2 = f[crossings], f[crossings + 1]
    x1, x2 = pc1[crossings], pc1[crossings + 1]
    fraction = f1.abs() / (f1.abs() + f2.abs())
    intersect_x = x1 + fraction * (x2 - x1)

    return intersect_x

# ==========================================
# 3. Feature Extraction
# ==========================================
def extract_features(intersections):
    """
    Extracts the 7 statistical features from the intersection points.
    """
    if intersections.numel() < 2:
        return torch.zeros(7, device=DEVICE)

    # 1. Range
    rng = intersections.max() - intersections.min()

    # 2. 0.13 Quantile
    q_013 = torch.quantile(intersections, 0.13)

    # 3. Interquartile Range (IQR)
    iqr = torch.quantile(intersections, 0.75) - torch.quantile(intersections, 0.25)

    # 4. Shannon Entropy (histogram-based density estimate)
    # torch.histc has no 'auto' bin rule like np.histogram, so bins are picked
    # via Sturges' rule as an approximation.
    n = intersections.numel()
    bins = max(1, int(torch.log2(torch.tensor(float(n))).ceil().item()) + 1)
    lo, hi = intersections.min(), intersections.max()
    counts = torch.histc(intersections, bins=bins, min=lo.item(), max=hi.item())
    bin_width = (hi - lo) / bins
    if bin_width > 0:
        density = counts / (counts.sum() * bin_width)
        density = density[density > 0]
        entropy = -(density * torch.log2(density)).sum()
    else:
        entropy = torch.tensor(0.0, device=DEVICE)

    # 5. Root Mean Squared Amplitude
    rms = torch.sqrt((intersections ** 2).mean())

    # 6. Coefficient of Variation
    mean_val = intersections.mean()
    cov = intersections.std(unbiased=False) / mean_val if mean_val != 0 else torch.tensor(0.0, device=DEVICE)

    # 7. Energy
    energy = (intersections ** 2).sum()

    return torch.stack([rng, q_013, iqr, entropy, rms, cov, energy])

# ==========================================
# 4. Multiprocessing Helper
# ==========================================
def extract_all_poincare_features(window_data):
    """
    Aplica a matemática do Poincaré em todos os canais e achata (flatten)
    em um tensor 1D para compatibilidade com o classificador PyTorch.
    """
    all_features = []
    for channel_signal in window_data:
        embedded = time_delay_embedding(channel_signal, d=5, tau=6)
        intersections = get_poincare_intersections(embedded)
        features = extract_features(intersections)
        all_features.append(features)
    return torch.cat(all_features)

# ==========================================
# 5. Processamento do Dataset
# ==========================================
def process_single_file(file_name, group, base_path, window_sec, sfreq):
    # Force joblib worker threads to suppress warnings locally
    import warnings
    warnings.filterwarnings('ignore')
    mne.set_log_level('ERROR')

    X_local, y_local = [], []
    patient = group['patient'].iloc[0]
    path_edf = os.path.join(base_path, patient, file_name)

    if not os.path.exists(path_edf):
        return None

    try:
        # 1. Carrega o arquivo permitindo nomes duplicados
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)

        # 2. Limpeza de nomes
        raw.rename_channels(lambda x: x.strip().upper())

        # 3. Mapeamento de sinônimos e Extração Direta
        existing_channels = raw.ch_names
        target_channels = [c.upper() for c in CHANNELS_TO_KEEP]
        selected_data = []

        for target in target_channels:
            if target in existing_channels:
                ch_idx = existing_channels.index(target)
                selected_data.append(raw.get_data(picks=ch_idx)[0])
            else:
                found_alt = [ch for ch in existing_channels if ch.startswith(target)]
                if found_alt:
                    ch_idx = existing_channels.index(found_alt[0])
                    selected_data.append(raw.get_data(picks=ch_idx)[0])

        # 4. Verifica se temos EXATAMENTE os canais necessários
        if len(selected_data) != len(target_channels):
            print(f"Erro em {file_name}: Encontrou {len(selected_data)} canais, mas precisava de {len(target_channels)}. Pulando arquivo.")
            return None

        # 5. Converte a lista em uma matriz NumPy e aplica a escala
        data = np.array(selected_data) * 1e6

        janela_samples = int(window_sec * sfreq)
        seizures = group[group['label'] == 1]
        is_seizure = np.zeros(data.shape[1], dtype=bool)

        # 6. Extract Seizure Windows (Label 1)
        for _, row in seizures.iterrows():
            start_idx = int(row['start_sec'] * sfreq)
            end_idx = int(row['end_sec'] * sfreq)

            is_seizure[start_idx:end_idx] = True

            for start in range(start_idx, end_idx - janela_samples, janela_samples):
                window = data[:, start:start + janela_samples]
                X_local.append(extract_all_poincare_features(window))
                y_local.append(1)

        # 7. Background (Label 0)
        cont_bg = 0
        while cont_bg < 15:
            start = np.random.randint(0, data.shape[1] - janela_samples)
            if not np.any(is_seizure[start: start + janela_samples]):
                window = data[:, start:start + janela_samples]
                X_local.append(extract_all_poincare_features(window))
                y_local.append(0)
                cont_bg += 1

        return torch.stack(X_local), torch.tensor(y_local, dtype=torch.long)

    except Exception as e:
        print(f"Erro em {file_name}: {e}")
        return None

if __name__ == "__main__":
    df = pd.read_csv('chb_mit_global_labels.csv')

    base_path = './dataset_chbmit'
    window_sec = 1
    sfreq = 256

    for patient_id, group in df.groupby('patient'):
        print(f"Processando características Poincaré (PyTorch) para {patient_id}...")

        results = Parallel(n_jobs=-1)(
            delayed(process_single_file)(f, file_group, base_path, window_sec, sfreq)
            for f, file_group in group.groupby('file_name')
        )

        X_patient, y_patient = [], []
        for res in results:
            if res is not None:
                if res[0].shape[0] > 0 and res[1].shape[0] > 0:
                    X_patient.append(res[0])
                    y_patient.append(res[1])

        if X_patient:
            X_stacked = torch.cat(X_patient, dim=0)
            y_stacked = torch.cat(y_patient, dim=0)
            torch.save(X_stacked, f'X_{patient_id}.pt')
            torch.save(y_stacked, f'y_{patient_id}.pt')
            print(f"Salvo {patient_id}: X={tuple(X_stacked.shape)}")
