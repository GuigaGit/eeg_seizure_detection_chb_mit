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
    Reconstructs the phase space of a 1D signal using time-delay embedding.

    The idea: instead of looking at the signal as a single scalar over time,
    we build a d-dimensional point for every time index t by taking the
    signal's value at t, t+tau, t+2*tau, ..., t+(d-1)*tau. Plotting these
    points traces out the trajectory of the underlying dynamical system in a
    reconstructed "phase space" (Takens' embedding theorem).

    How: for each of the d "delay" offsets (0, tau, 2*tau, ...), we take a
    shifted slice of the signal of length `valid_length` and stack the d
    slices as columns, so row i of the output is
    [signal[i], signal[i+tau], ..., signal[i+(d-1)*tau]].

    Args:
        signal: 1D array-like (numpy array, list, or torch tensor) with the
            raw time-domain samples of one EEG channel/window.
        d: embedding dimension - how many delayed copies of the signal are
            stacked together (5 in the reference paper).
        tau: time lag between consecutive copies, in samples (6 samples,
            i.e. ~23ms at 256Hz, in the reference paper).

    Returns:
        A torch.Tensor of shape (valid_length, d), where
        valid_length = len(signal) - (d - 1) * tau. Each row is one point of
        the reconstructed trajectory. If the signal is too short for the
        chosen d/tau, returns a single zero row of shape (1, d) instead of
        raising, so batch processing pipelines don't crash on edge windows.
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
    Projects points onto their principal components (PyTorch replacement for
    sklearn.decomposition.PCA().fit_transform()).

    How: PCA is computed via SVD of the mean-centered data instead of via an
    eigendecomposition of the covariance matrix - numerically this is the
    same result (the right singular vectors Vh are the principal directions),
    but avoids ever forming the (d x d) covariance matrix explicitly. The
    directions are already sorted by descending explained variance (largest
    singular value first), matching sklearn's convention: column 0 of the
    output is "PC1", column 1 is "PC2", etc.

    Args:
        embedded: torch.Tensor of shape (n_points, d) - e.g. the output of
            `time_delay_embedding`. Each row is one observation, each column
            one original dimension.
        n_components: how many leading principal components to keep. If
            None, all d components are returned (used by
            visualize_pipeline.py to inspect PC1/PC2/PC3 together).

    Returns:
        torch.Tensor of shape (n_points, n_components) (or (n_points, d) if
        n_components is None) with the data expressed in principal-component
        coordinates.
    """
    mean = embedded.mean(dim=0, keepdim=True)
    centered = embedded - mean
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    if n_components is not None:
        Vh = Vh[:n_components]
    return centered @ Vh.T


def fit_line(x, y):
    """
    Fits the best-fit straight line y = m*x + c through a set of 2D points,
    in the least-squares sense (torch equivalent of np.polyfit(x, y, 1)).

    How: builds the design matrix A = [x, 1] (one row per point) and solves
    the linear system A @ [m, c]^T ≈ y for the [m, c] that minimizes the sum
    of squared residuals, using `torch.linalg.lstsq` (QR-based least squares,
    same idea as normal equations but numerically more stable).

    Args:
        x: 1D torch.Tensor of x-coordinates (e.g. PC1 values).
        y: 1D torch.Tensor of y-coordinates (e.g. PC2 values), same length
            as x.

    Returns:
        Tuple (m, c) of 0-dimensional torch.Tensors: m is the slope, c is
        the intercept of the fitted line.
    """
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    return solution[0, 0], solution[1, 0]


def get_poincare_intersections(embedded_space):
    """
    Computes the Poincaré section of a reconstructed trajectory: the points
    where the trajectory crosses a fixed reference line, which is a standard
    way to turn a continuous chaotic trajectory into a discrete sequence of
    numbers that's easier to summarize statistically.

    How, step by step:
      1. Reduce the d-dimensional embedded trajectory to 2D via PCA (using
         `pca_transform`), giving coordinates (pc1, pc2) per point.
      2. Fit a straight line through the 2D trajectory with `fit_line` - this
         line is the "Poincaré section" (the cutting plane, projected to 2D).
      3. For each consecutive pair of trajectory points, compute the signed
         vertical distance to the line, f = pc2 - (m*pc1 + c). If f changes
         sign between one point and the next, the trajectory crossed the
         line between them.
      4. For each such crossing, linearly interpolate between the two points
         (weighted by how close each one's f is to zero) to estimate the
         exact PC1 value where the crossing happened.
      This sign-crossing search is fully vectorized with tensor ops instead
      of a Python for-loop (unlike the original SciPy version), which is
      both faster and GPU-friendly.

    Args:
        embedded_space: torch.Tensor of shape (n_points, d), typically the
            output of `time_delay_embedding` for one signal window.

    Returns:
        1D torch.Tensor of the PC1 coordinate at every point where the
        trajectory crosses the fitted line, in the order the crossings occur
        (i.e. a discrete time series of "Poincaré return values"). Returns
        an empty tensor if there aren't enough points, the input is flat
        (zero variance - nothing to embed), or there are no crossings at all.
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
    Summarizes a (potentially long) sequence of Poincaré intersection values
    into a fixed-length vector of 7 statistics, so that windows of any
    duration produce the same number of features and can be fed to a
    classifier.

    The 7 features, in order, and what each one captures about the spread /
    shape of the intersection values' distribution:
      1. Range: max - min (overall spread).
      2. 0.13 quantile: an asymmetric, robust "low" percentile (chosen in
         the reference paper; not the median, so it's sensitive to skew).
      3. Interquartile range (IQR = 75th - 25th percentile): spread that
         ignores outliers, unlike Range.
      4. Shannon entropy: how "spread out"/unpredictable the distribution of
         values is, estimated from a histogram (density estimate) of the
         intersections; bin count follows Sturges' rule since torch has no
         'auto' bin-width heuristic like numpy.
      5. RMS (root-mean-square): overall amplitude/energy scale, but as an
         average rather than a sum, so it doesn't grow with more crossings.
      6. Coefficient of variation: std / mean, a scale-free measure of
         relative variability (0 if the mean is 0, to avoid a divide-by-zero).
      7. Energy: sum of squares - grows with the number of crossings, unlike
         RMS, so it also implicitly encodes "how many crossings happened".

    Args:
        intersections: 1D torch.Tensor of Poincaré intersection values, as
            returned by `get_poincare_intersections` for one channel/window.

    Returns:
        1D torch.Tensor of length 7 with the features above, in the order
        listed. If there are fewer than 2 intersections (not enough data to
        compute spread statistics meaningfully), returns a zero vector
        instead of NaNs, so downstream stacking/training never breaks.
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
    Runs the full per-channel Poincaré feature pipeline (embed -> intersect
    -> summarize) on every channel of a multi-channel EEG window, and
    concatenates the results into one flat feature vector.

    How: for each channel's 1D signal, it chains
    `time_delay_embedding` -> `get_poincare_intersections` -> `extract_features`
    to get 7 numbers per channel, then concatenates all channels' 7-number
    blocks end to end (so the feature at index [7*ch : 7*(ch+1)] belongs to
    channel `ch`). This flat 1D shape is what the linear classifiers in
    `svm_training.py` and `pca_svm_pipeline.py` expect as input.

    Args:
        window_data: array-like of shape (n_channels, n_samples) - one short
            EEG window (e.g. 1 second) with all channels for that window.
            Each row is one channel's raw signal.

    Returns:
        1D torch.Tensor of length 7 * n_channels: the concatenation of the
        7-feature vector for every channel, in the same channel order as the
        input.
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
    """
    Builds the labeled Poincaré-feature dataset for a single .edf recording:
    loads it, extracts short seizure windows and an equal-ish sample of
    background windows, and turns each window into a feature vector.

    How, step by step:
      1. Load the raw .edf file with MNE and standardize channel names
         (strip whitespace, uppercase) so they can be matched against the
         fixed `CHANNELS_TO_KEEP` montage regardless of how this particular
         recording's channels happen to be named/ordered.
      2. Select exactly the channels in `CHANNELS_TO_KEEP` (falling back to
         a channel whose name starts with the target, to tolerate suffixes
         like "T8-P8-1"); if any target channel is simply missing from this
         file, the whole file is skipped (returns None) rather than
         producing a dataset with an inconsistent number of channels.
      3. Convert from Volts to microvolts (the paper's feature scale).
      4. For every annotated seizure interval in `group`, slice it into
         non-overlapping `window_sec`-long windows, extract Poincaré
         features per window (`extract_all_poincare_features`), and label
         them 1.
      5. Sample 15 background windows at random positions that don't
         overlap any seizure interval, and label them 0 - a fixed count per
         file so background doesn't overwhelm the (rarer) seizure windows.

    Args:
        file_name: name of the .edf file to process (e.g. "chb01_03.edf").
        group: pandas DataFrame slice (rows of `chb_mit_global_labels.csv`
            for this file) with columns 'patient', 'label', 'start_sec',
            'end_sec' describing the seizure intervals annotated in it.
        base_path: root folder containing one subfolder per patient
            (typically `DATASET_DIR`), used to locate the .edf file.
        window_sec: length of each feature-extraction window, in seconds.
        sfreq: sampling frequency of the recording, in Hz (used to convert
            seconds to sample indices).

    Returns:
        Tuple (X_local, y_local) where X_local is a torch.Tensor of shape
        (n_windows, 7 * n_channels) stacking one feature vector per window,
        and y_local is a torch.Tensor of shape (n_windows,) with the
        matching 0/1 labels. Returns None if the .edf file doesn't exist,
        doesn't contain all required channels, or fails to load/process for
        any other reason (the error is printed, not raised, so a single bad
        file doesn't kill the whole parallel batch in the __main__ block
        below).
    """
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
    df = pd.read_csv(os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv'))

    base_path = DATASET_DIR
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
            torch.save(X_stacked, os.path.join(SCRIPT_DIR, f'X_{patient_id}.pt'))
            torch.save(y_stacked, os.path.join(SCRIPT_DIR, f'y_{patient_id}.pt'))
            print(f"Salvo {patient_id}: X={tuple(X_stacked.shape)}")
