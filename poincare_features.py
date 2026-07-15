import mne
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import warnings

# This forces MNE to only print fatal errors, hiding the warnings
mne.set_log_level('ERROR') 
warnings.filterwarnings('ignore')

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
    Reconstructs the phase space using time-delay embedding.
    """
    n_samples = len(signal)
    valid_length = n_samples - (d - 1) * tau
    
    if valid_length <= 0:
        # Fallback to prevent crashes on extremely short signal edges
        return np.zeros((1, d)) 
        
    embedded = np.zeros((valid_length, d))
    for i in range(d):
        embedded[:, i] = signal[i*tau : valid_length + i*tau]
        
    return embedded

# ==========================================
# 2. PCA & Poincaré Section Mapping
# ==========================================
def get_poincare_intersections(embedded_space):
    """
    Applies PCA, fits a 1st-degree polynomial (line), and finds intersections.
    Returns the PC1 values of the intersection points.
    """
    if embedded_space.shape[0] < 2:
        return np.array([])
        
    # Check variance BEFORE PCA to avoid divide-by-zero warnings on flat signals
    if np.var(embedded_space) <= 1e-12:
        return np.array([])
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embedded_space)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    
    # Protect against flat-line signals (zero variance)
    if np.var(pc1) == 0:
        return np.array([])
    
    # Usa uma solucao least-squares ao inves do Bezier Clipping para encontrar a linha de melhor ajuste
    # So pq eu ainda nao entendi como fazer o Bezier Clipping direito. Mas a ideia é a mesma: encontrar a linha que melhor separa os pontos
    m, c = np.polyfit(pc1, pc2, 1)
    intersection_pc1_values = []
    
    for i in range(len(pc1) - 1):
        x1, y1 = pc1[i], pc2[i]
        x2, y2 = pc1[i+1], pc2[i+1]
        
        # Calculates the vertical distance from the Poincaré line to Point 1 (f1) and Point 2 (f2). 
        # If a point is above the line, f is positive. If it is below, f is negative.
        f1 = y1 - (m * x1 + c)
        f2 = y2 - (m * x2 + c)
        
        # If multiply f1 and f2 and the result is negative, it guarantees one point was positive (above) and one was negative (below). 
        # This proves the trajectory just pierced the Poincaré section
        if f1 * f2 < 0:
            m2 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
            c2 = y1 - m2 * x1
            intersect_x = (c2 - c) / (m - m2) if (m - m2) != 0 else np.nan
            intersection_pc1_values.append(intersect_x)
            
    return np.array(intersection_pc1_values)

# ==========================================
# 3. Feature Extraction
# ==========================================
def extract_features(intersections):
    """
    Extracts the 7 statistical features from the intersection points.
    """
    if len(intersections) < 2:
        return np.zeros(7)
    
    # 1. Range
    rng = np.max(intersections) - np.min(intersections)

    # 2. 0.13 Quantile
    q_013 = np.quantile(intersections, 0.13)

    # 3. Interquartile Range (IQR)
    iqr = np.percentile(intersections, 75) - np.percentile(intersections, 25)
    
    # 4. Shannon Entropy
    # We estimate the PDF using a histogram
    hist, _ = np.histogram(intersections, density=True, bins='auto')
    hist = hist[hist > 0] 
    entropy = -np.sum(hist * np.log2(hist))
    
    # 5. Root Mean Squared Amplitude
    rms = np.sqrt(np.mean(intersections**2))
    
    # 6. Coefficient of Variation
    mean_val = np.mean(intersections)
    cov = np.std(intersections) / mean_val if mean_val != 0 else 0
    
    # 7. Energy
    energy = np.sum(intersections**2)
    
    return np.array([rng, q_013, iqr, entropy, rms, cov, energy])

# ==========================================
# 4. Multiprocessing Helper
# ==========================================
def extract_all_poincare_features(window_data):
    """
    Substitui a antiga extract_all_features. 
    Aplica a matemática do Poincaré em todos os canais e achata (flatten) 
    em um array 1D para compatibilidade com o SVM.
    """
    all_features = []
    for channel_signal in window_data:
        embedded = time_delay_embedding(channel_signal, d=5, tau=6)
        intersections = get_poincare_intersections(embedded)
        features = extract_features(intersections)
        all_features.extend(features)
    return np.array(all_features)

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
    
    if not os.path.exists(path_edf): return None
    
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
                window = data[:, start:start+janela_samples]
                X_local.append(extract_all_poincare_features(window))
                y_local.append(1)
        
        # 7. Background (Label 0)
        cont_bg = 0
        while cont_bg < 15:
            start = np.random.randint(0, data.shape[1] - janela_samples)
            if not np.any(is_seizure[start : start + janela_samples]):
                window = data[:, start:start+janela_samples]
                X_local.append(extract_all_poincare_features(window))
                y_local.append(0)
                cont_bg += 1
        
        return np.array(X_local), np.array(y_local)
    
    except Exception as e:
        print(f"Erro em {file_name}: {e}")
        return None

if __name__ == "__main__":
    df = pd.read_csv('chb_mit_global_labels.csv')

    base_path = './dataset_chbmit'
    window_sec = 1 
    sfreq = 256
    
    for patient_id, group in df.groupby('patient'):
        print(f"Processando características Poincaré para {patient_id}...")
        
        results = Parallel(n_jobs=-1)(
            delayed(process_single_file)(f, file_group, base_path, window_sec, sfreq) 
            for f, file_group in group.groupby('file_name')
        )
        
        X_patient, y_patient = [], []
        for res in results:
            if res is not None:
                # Ensure the local arrays are not empty before appending
                if len(res[0]) > 0 and len(res[1]) > 0:
                    X_patient.append(res[0])
                    y_patient.append(res[1])
                
        if X_patient:
            X_stacked = np.vstack(X_patient)
            y_stacked = np.concatenate(y_patient)
            np.save(f'X_{patient_id}.npy', X_stacked)
            np.save(f'y_{patient_id}.npy', y_stacked)
            print(f"Salvo {patient_id}: X={X_stacked.shape}")