import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from poincare_features import time_delay_embedding, get_poincare_intersections

# Desativar logs excessivos do MNE
mne.set_log_level('ERROR')

def load_real_segments(base_path='./dataset_chbmit'):
    """
    Busca no chb01 um arquivo com crise e extrai uma janela de 1s de background
    e uma janela de 1s de crise para fins de comparação visual.
    """
    # Carregando informações globais de labels para o chb01
    df = pd.read_csv('chb_mit_global_labels.csv')
    df_patient = df[df['patient'] == 'chb01']
    
    # Encontra um arquivo que possua crise anotada
    seizure_file_info = df_patient[df_patient['label'] == 1].iloc[0]
    file_name = seizure_file_info['file_name']
    start_seizure = seizure_file_info['start_sec']
    
    edf_path = os.path.join(base_path, 'chb01', file_name)
    print(f"Carregando arquivo EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Selecionar canal representativo (ex: FP1-F7)
    channel_name = 'P7-T7'
    raw.pick_channels([channel_name])
    data, times = raw[:, :]
    signal = data[0]
    sfreq = int(raw.info['sfreq'])
    
    # 1 segundo de sinal = 256 amostras
    # Janela normal (longe do início da crise)
    # segundo 5 apos o início da gravação para evitar artefatos iniciais
    bg_start_idx = int(5 * sfreq)
    bg_signal = signal[bg_start_idx : bg_start_idx + sfreq]
    
    # Janela de crise (exatamente no início da crise anotada)
    sz_start_idx = int(3026 * sfreq)
    sz_signal = signal[sz_start_idx : sz_start_idx + sfreq]
    
    return bg_signal, sz_signal, sfreq

def plot_pipeline_validation(signal, title_prefix="Segmento"):
    """
    Gera o plot de 4 etapas para validação matemática do bloco geométrico,
    incluindo a sequência temporal discreta de interseções.
    """
    # 1. Reconstrução do Espaço de Fase 5D
    embedded = time_delay_embedding(signal, d=5, tau=6)
    
    # 2. PCA para redução de dimensionalidade (5D -> 2D/3D)
    pca = PCA(n_components=None) # Ajusta todos os componentes
    pca.fit(embedded)
    coords_reduced = pca.transform(embedded)
    
    # Usando os três componentes principais mais proeminentes
    pc1 = coords_reduced[:, 0]
    pc2 = coords_reduced[:, 1]
    pc3 = coords_reduced[:, 2] if coords_reduced.shape[1] > 2 else np.zeros_like(pc1)
    
    # Ajuste polinomial de grau 1 (a Seção de Poincaré)
    coefficients = np.polyfit(pc1, pc2, 1)
    polynomial = np.poly1d(coefficients)
    
    x_line = np.linspace(min(pc1), max(pc1), 100)
    y_line = polynomial(x_line)
    
    # 3. Encontrar interseções
    intersections_pc1 = get_poincare_intersections(embedded)
    
    # --- Configuração do Plot (Grade 2x2) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Validação Completa do Pipeline - {title_prefix}", fontsize=16, fontweight='bold')
    
    # Subplot [0, 0]: Sinal de EEG Bruto
    axs[0, 0].plot(np.arange(len(signal)) / 256.0, signal, color='darkblue', lw=1.5)
    axs[0, 0].set_title("1. Sinal de EEG de 1 segundo")
    axs[0, 0].set_xlabel("Tempo (s)")
    axs[0, 0].set_ylabel("Amplitude (uV)")
    axs[0, 0].grid(True, alpha=0.5)
    
    # Subplot [0, 1]: Trajetória Espaço de Fase 3D
    # Removemos o subplot padrão 2D e adicionamos uma projeção 3D nele
    pos = axs[0, 1].get_position()
    axs[0, 1].remove()
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_3d.set_position(pos)
    ax_3d.plot3D(pc1, pc2, pc3, color='purple', alpha=0.8, lw=1)
    ax_3d.scatter3D(pc1[0], pc2[0], pc3[0], color='green', s=40, label='Início')
    ax_3d.scatter3D(pc1[-1], pc2[-1], pc3[-1], color='red', s=40, label='Fim')
    ax_3d.set_title("2. Trajetória Reduzida (PCs 1, 2, 3)")
    ax_3d.set_xlabel("PC 1")
    ax_3d.set_ylabel("PC 2")
    ax_3d.set_zlabel("PC 3")
    ax_3d.legend()
    
    # Subplot [1, 0]: Seção de Poincaré 2D e Pontos de Cruzamento
    axs[1, 0].plot(pc1, pc2, color='gray', alpha=0.6, linestyle='--', label='Trajetória 2D')
    axs[1, 0].plot(x_line, y_line, color='red', lw=2, label='Seção de Poincaré (Linha)')
    # Como as interseções representam os valores de PC1 nos cruzamentos:
    intersections_pc2 = polynomial(intersections_pc1)
    axs[1, 0].scatter(intersections_pc1, intersections_pc2, color='gold', edgecolor='black', s=50, zorder=5, label='Interseções')
    axs[1, 0].set_title("3. Seção de Poincaré e Cruzamentos")
    axs[1, 0].set_xlabel("PC 1")
    axs[1, 0].set_ylabel("PC 2")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    # Subplot [1, 1]: Sequência Temporal Discreta das Interseções (O gráfico que faltava!)
    indices_intersecoes = np.arange(1, len(intersections_pc1) + 1)
    axs[1, 1].plot(indices_intersecoes, intersections_pc1, color='teal', linestyle='-', marker='o', 
                  markerfacecolor='gold', markeredgecolor='black', markersize=6, lw=1.5, label='Valor da Interseção ($x_I$)')
    axs[1, 1].set_title("4. Sequência de Interseções de Poincaré")
    axs[1, 1].set_xlabel("Índice do Cruzamento ($n$)")
    axs[1, 1].set_ylabel("Valor Projetado no PC 1 ($x_I(n)$)")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()
    
    plt.tight_layout()
    os.makedirs('results/pipeline_validation', exist_ok=True)
    save_path = f"results/pipeline_validation/{title_prefix.lower().replace(' ', '_')}_validation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Gráfico completo salvo com sucesso em: {save_path}")
    plt.show()

if __name__ == "__main__":
    try:
        # Carrega dados reais do chb01 do seu dataset local
        normal_segment, seizure_segment, sfreq = load_real_segments()
        
        # Gera o plot de validação para o sinal normal
        plot_pipeline_validation(normal_segment, title_prefix="Segmento Normal (Background)")
        
        # Gera o plot de validação para o sinal de crise
        plot_pipeline_validation(seizure_segment, title_prefix="Segmento de Crise (Ictal)")
        
    except FileNotFoundError as e:
        print(f"\n[AVISO] Não foi possível encontrar os arquivos de dados reais do CHB-MIT: {e}")