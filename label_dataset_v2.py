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
    sfreq = 256 

    # Canais padrão do sistema 10-20 presentes no CHB-MIT
    channels_to_keep = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]

    for file_name, group in df.groupby('file_name'):
        patient = group['patient'].iloc[0]
        path_edf = os.path.join(base_path, patient, file_name)
        
        if not os.path.exists(path_edf): continue
        
        print(f"Lendo: {file_name}")
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)
        
        # Tenta selecionar apenas os canais padrão. Ignora arquivos que não os possuam.
        try:
            raw.pick_channels(channels_to_keep)
        except ValueError:
            print(f"Aviso: Arquivo {file_name} não possui a montagem padrão. Pulando...")
            continue

        raw.filter(0.5, 40, verbose=False)
        raw.notch_filter(60, verbose=False)
        data = raw.get_data()
        
        # 1. Extrair Janelas de Crise (Label 1)
        seizures = group[group['label'] == 1]
        for _, row in seizures.iterrows():
            s_idx, e_idx = int(row['start_sec']*sfreq), int(row['end_sec']*sfreq)
            duracao_samples = e_idx - s_idx
            janela_samples = int(window_sec * sfreq)

            # Se a crise for mais curta que a janela, pegamos a janela começando no início da crise
            if duracao_samples < janela_samples:
                # Garante que não ultrapasse o fim do arquivo
                if s_idx + janela_samples < data.shape[1]:
                    win = data[:, s_idx : s_idx + janela_samples]
                    X.append(extract_all_features(win, sfreq))
                    y.append(1)
            else:
                for i in range(s_idx, e_idx - janela_samples, janela_samples):
                    win = data[:, i : i + janela_samples]
                    X.append(extract_all_features(win, sfreq))
                    y.append(1)

    return np.array(X), np.array(y)

X, y = build_complete_dataset('./dataset', 'chb_mit_global_labels.csv')
np.save('X_final.npy', X)
np.save('y_final.npy', y)