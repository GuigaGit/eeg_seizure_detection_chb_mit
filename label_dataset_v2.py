import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def extract_all_features(window_data, sfreq):
    """
    Extrai um conjunto abrangente de características para cada canal.
    window_data: shape (23, samples)
    """
    all_features = []
    
    for channel_signal in window_data:
        # --- DOMÍNIO DO TEMPO ---
        # 1-4. Estatísticas Básicas
        all_features.append(np.mean(channel_signal))
        all_features.append(np.var(channel_signal))
        all_features.append(skew(channel_signal))
        all_features.append(kurtosis(channel_signal))
        
        # 5. Root Mean Square (RMS) - Medida de potência no tempo
        all_features.append(np.sqrt(np.mean(channel_signal**2)))
        
        # 6. Line Length - Crucial para detectar espículas rítmicas de crises
        line_length = np.sum(np.abs(np.diff(channel_signal)))
        all_features.append(line_length)
        
        # --- DOMÍNIO DA FREQUÊNCIA ---
        # 7-11. Potência absoluta em bandas clássicas (Welch)
        freqs, psd = welch(channel_signal, sfreq, nperseg=sfreq) # Resolução de 1Hz
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        
        for fmin, fmax in bands:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            all_features.append(np.trapz(psd[idx], freqs[idx])) # Área sob a curva (potência)
            
    return np.array(all_features)

def build_complete_dataset(base_path, global_labels_csv, window_sec=4):
    df = pd.read_csv(global_labels_csv)
    X, y = [], []
    sfreq = 256 # Definido no sumário 

    # Para evitar carregar o mesmo EDF várias vezes se houver múltiplas crises
    for file_name, group in df.groupby('file_name'):
        patient = group['patient'].iloc[0]
        path_edf = os.path.join(base_path, patient, file_name)
        
        if not os.path.exists(path_edf): continue
        
        print(f"Processando: {file_name}")
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)
        raw.filter(0.5, 40, verbose=False) # Recomendação clínica
        raw.notch_filter(60, verbose=False) # Notch local
        data = raw.get_data()
        
        # 1. Extrair Janelas de Crise (Label 1)
        seizures = group[group['label'] == 1]
        for _, row in seizures.iterrows():
            s_idx, e_idx = int(row['start_sec']*sfreq), int(row['end_sec']*sfreq)
            for i in range(s_idx, e_idx - int(window_sec*sfreq), int(window_sec*sfreq)):
                win = data[:, i : i + int(window_sec*sfreq)]
                X.append(extract_all_features(win, sfreq))
                y.append(1)
        
        # 2. Extrair Janelas de Não-Crise (Label 0) - Amostragem para balancear
        # Se o arquivo não tem crise, pegamos algumas janelas aleatórias
        if len(seizures) == 0:
            # Pega 10 janelas aleatórias por arquivo sem crise para não sobrecarregar
            for _ in range(10):
                start = np.random.randint(0, data.shape[1] - int(window_sec*sfreq))
                win = data[:, start : start + int(window_sec*sfreq)]
                X.append(extract_all_features(win, sfreq))
                y.append(0)

    return np.array(X), np.array(y)

X, y = build_complete_dataset('./dataset', 'chb_mit_global_labels.csv')
np.save('X_final.npy', X)
np.save('y_final.npy', y)