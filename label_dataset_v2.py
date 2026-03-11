import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def extract_all_features(window_data, sfreq):
    """
    Extrai um conjunto abrangente de características para cada canal.
    window_data: shape (canais, amostras)
    """
    all_features = []
    
    for channel_signal in window_data:
        # --- DOMÍNIO DO TEMPO ---
        # Estatísticas Básicas
        all_features.append(np.mean(channel_signal))
        all_features.append(np.var(channel_signal))
        all_features.append(skew(channel_signal))
        all_features.append(kurtosis(channel_signal))
        
        # Root Mean Square (RMS)
        all_features.append(np.sqrt(np.mean(channel_signal**2)))
        
        # Line Length
        line_length = np.sum(np.abs(np.diff(channel_signal)))
        all_features.append(line_length)
        
        # --- DOMÍNIO DA FREQUÊNCIA ---
        # Potência absoluta em bandas clássicas (Welch)
        freqs, psd = welch(channel_signal, sfreq, nperseg=sfreq) 
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        
        for fmin, fmax in bands:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            all_features.append(np.trapz(psd[idx], freqs[idx])) 
            
    return np.array(all_features)

def build_complete_dataset(base_path, global_labels_csv, window_sec=4):
    df = pd.read_csv(global_labels_csv)
    X, y = [], []
    sfreq = 256 

    # Canais padrão do sistema 10-20 (Montagem comum no CHB-MIT)
    channels_to_keep = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]

    for file_name, group in df.groupby('file_name'):
        patient = group['patient'].iloc[0]
        path_edf = os.path.join(base_path, patient, file_name)
        
        if not os.path.exists(path_edf): 
            continue
        
        print(f"Processando: {file_name}")
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)
        
        # COMMIT 2: Força a mesma quantidade de canais em todos os arquivos
        try:
            raw.pick_channels(channels_to_keep)
        except ValueError:
            print(f"Aviso: Arquivo {file_name} ignorado por falta de canais padrão.")
            continue

        raw.filter(0.5, 40, verbose=False) 
        raw.notch_filter(60, verbose=False) 
        data = raw.get_data()
        janela_samples = int(window_sec * sfreq)
        
        # 1. Extrair Janelas de Crise (Label 1)
        seizures = group[group['label'] == 1]
        for _, row in seizures.iterrows():
            s_idx, e_idx = int(row['start_sec']*sfreq), int(row['end_sec']*sfreq)
            duracao_samples = e_idx - s_idx

            # Garante captura de crises curtas (menores que a janela)
            if duracao_samples < janela_samples:
                if s_idx + janela_samples < data.shape[1]:
                    win = data[:, s_idx : s_idx + janela_samples]
                    X.append(extract_all_features(win, sfreq))
                    y.append(1)
            else:
                for i in range(s_idx, e_idx - janela_samples, janela_samples):
                    win = data[:, i : i + janela_samples]
                    X.append(extract_all_features(win, sfreq))
                    y.append(1)
        
        # 2. Extrair Janelas de Não-Crise (Label 0)
        # Amostragem de fundo em todos os arquivos com máscara anti-sobreposição
        total_samples = data.shape[1]
        is_seizure = np.zeros(total_samples, dtype=bool)
        for _, row in seizures.iterrows():
            is_seizure[int(row['start_sec']*sfreq) : int(row['end_sec']*sfreq)] = True
        
        cont_bg = 0
        tentativas = 0
        while cont_bg < 15 and tentativas < 100:
            start = np.random.randint(0, total_samples - janela_samples)
            if not np.any(is_seizure[start : start + janela_samples]):
                win = data[:, start : start + janela_samples]
                X.append(extract_all_features(win, sfreq))
                y.append(0)
                cont_bg += 1
            tentativas += 1

    return np.array(X), np.array(y)

# Execução Principal
# Certifique-se de que o caminho './dataset_chbmit' existe e contém as pastas chb01, chb02...
X, y = build_complete_dataset('./dataset_chbmit', 'chb_mit_global_labels.csv')
np.save('X_final.npy', X)
np.save('y_final.npy', y)
print(f"Dataset finalizado! Formato de X: {X.shape}, Formato de y: {y.shape}")