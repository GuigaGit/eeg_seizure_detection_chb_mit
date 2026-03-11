import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

# Canais padrão do sistema 10-20 (Montagem comum no CHB-MIT)
CHANNELS_TO_KEEP = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

def extract_all_features(window_data, sfreq):
    all_features = []
    for channel_signal in window_data:
        # Domínio do Tempo
        all_features.append(np.mean(channel_signal))
        all_features.append(np.var(channel_signal))
        all_features.append(skew(channel_signal))
        all_features.append(kurtosis(channel_signal))
        all_features.append(np.sqrt(np.mean(channel_signal**2)))
        all_features.append(np.sum(np.abs(np.diff(channel_signal))))
        
        # Domínio da Frequência (Bandas de EEG)
        freqs, psd = welch(channel_signal, sfreq, nperseg=sfreq) 
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        for fmin, fmax in bands:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            all_features.append(np.trapz(psd[idx], freqs[idx])) 
    return np.array(all_features)

def process_single_file(file_name, group, base_path, window_sec, sfreq):
    X_local, y_local = [], []
    patient = group['patient'].iloc[0]
    path_edf = os.path.join(base_path, patient, file_name)
    
    if not os.path.exists(path_edf): return None
    
    try:
        # 1. Carrega o arquivo permitindo nomes duplicados (ele vai renomear automaticamente)
        raw = mne.io.read_raw_edf(path_edf, preload=True, verbose=False)
        
        # 2. Limpeza de nomes: remove espaços e converte para maiúsculas para facilitar a comparação
        raw.rename_channels(lambda x: x.strip().upper())
        
        # 3. Mapeamento de sinonimos comuns no CHB-MIT (Ex: T8-P8 as vezes vem com -1 ou .1)
        # Vamos tentar encontrar os canais que REALMENTE existem no arquivo
        existing_channels = raw.ch_names
        final_selection = []
        
        # Lista de canais desejados (em maiúsculas para bater com o rename acima)
        target_channels = [c.upper() for c in CHANNELS_TO_KEEP]
        
        for target in target_channels:
            if target in existing_channels:
                final_selection.append(target)
            else:
                # Tenta buscar variações comuns (ex: T8-P8-1)
                found_alt = [ch for ch in existing_channels if ch.startswith(target)]
                if found_alt:
                    final_selection.append(found_alt[0])

        # 4. Verifica se temos canais suficientes para treinar (ex: pelo menos 15 dos 18)
        if len(final_selection) < 15:
            print(f"Erro em {file_name}: Apenas {len(final_selection)} canais encontrados. Pulando.")
            return None
            
        # 5. Usa o novo método .pick (o pick_channels é legado)
        raw.pick(final_selection)
        
        # 6. Garante que agora todos tenham EXATAMENTE os mesmos nomes para o NumPy não reclamar
        # Renomeia de volta para os nomes originais da sua lista CHANNELS_TO_KEEP
        rename_dict = {actual: original for actual, original in zip(raw.ch_names, CHANNELS_TO_KEEP)}
        raw.rename_channels(rename_dict)
        
        # 2. Background (Label 0) - Amostragem robusta
        is_seizure = np.zeros(data.shape[1], dtype=bool)
        for _, row in seizures.iterrows():
            is_seizure[int(row['start_sec']*sfreq) : int(row['end_sec']*sfreq)] = True
        
        cont_bg = 0
        while cont_bg < 15:
            start = np.random.randint(0, data.shape[1] - janela_samples)
            if not np.any(is_seizure[start : start + janela_samples]):
                X_local.append(extract_all_features(data[:, start:start+janela_samples], sfreq))
                y_local.append(0)
                cont_bg += 1
        return np.array(X_local), np.array(y_local)
    except Exception as e:
        print(f"Erro em {file_name}: {e}")
        return None

def build_complete_dataset(base_path, global_labels_csv, window_sec=4):
    df = pd.read_csv(global_labels_csv)
    sfreq = 256
    
    # Execução em Paralelo (n_jobs=-1 usa todos os núcleos)
    results = Parallel(n_jobs=-1)(
        delayed(process_single_file)(f, g, base_path, window_sec, sfreq) 
        for f, g in df.groupby('file_name')
    )
    
    X, y = [], []
    for res in results:
        if res is not None:
            X.append(res[0])
            y.append(res[1])
            
    return np.vstack(X), np.concatenate(y)

if __name__ == "__main__":
    X, y = build_complete_dataset('./dataset_chbmit', 'chb_mit_global_labels.csv')
    np.save('X_final.npy', X)
    np.save('y_final.npy', y)
    print(f"Processamento concluído. Dataset: {X.shape}")