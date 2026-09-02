import pandas as pd
import re
import os
import sys

# Este script apenas faz parsing de texto (sem PCA/SVM/nenhum modelo), então
# não há nada "PyTorch" para trocar aqui - mantido idêntico ao original para
# que o pipeline em PyTorch/ continue autocontido.

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

def extrair_dados_sumario(caminho_arquivo):
    """
    Parses one CHB-MIT "-summary.txt" file into a list of seizure/background
    label records, one per .edf recording described in it. This is pure
    text parsing (no signal processing, no model) - it just turns the
    dataset's free-text summary format into structured rows that the rest
    of the pipeline (poincare_features.py, label_dataset_v2.py) can join
    against actual recordings by file name.

    How: splits the summary file's text on "File Name: " (each chunk after
    the first is one recording's block of text), then for each block:
      - reads the file name off the first line;
      - regex-searches for "Number of Seizures in File: N";
      - if N > 0, regex-finds every "Seizure ... Start/End Time: X seconds"
        pair in the block and emits one label=1 record per seizure, with
        its start/end in seconds;
      - if N == 0, emits a single label=0 record with start_sec=end_sec=0
        (a placeholder, since there's no seizure interval to report - the
        downstream code only cares that label 0 means "no seizure in this
        file").

    Args:
        caminho_arquivo: path to one patient's "-summary.txt" file.

    Returns:
        List of dicts, each with keys 'file_name', 'start_sec', 'end_sec',
        'label' (1 for a seizure interval, 0 for a seizure-free file) - one
        dict per seizure interval, or one dict per seizure-free file.
    """
    with open(caminho_arquivo, 'r') as f:
        conteudo = f.read()

    blocos = conteudo.split('File Name: ')
    lista_dados = []

    for bloco in blocos[1:]:
        linhas = bloco.split('\n')
        nome_arquivo = linhas[0].strip()

        match_crises = re.search(r'Number of Seizures in File:\s*(\d+)', bloco)
        num_crises = int(match_crises.group(1)) if match_crises else 0

        if num_crises > 0:
            starts = re.findall(r'Seizure (?:\d+ )?Start Time:\s*(\d+)\s*seconds', bloco)
            ends = re.findall(r'Seizure (?:\d+ )?End Time:\s*(\d+)\s*seconds', bloco)

            for s, e in zip(starts, ends):
                lista_dados.append({
                    'file_name': nome_arquivo,
                    'start_sec': int(s),
                    'end_sec': int(e),
                    'label': 1
                })
        else:
            lista_dados.append({
                'file_name': nome_arquivo,
                'start_sec': 0,
                'end_sec': 0,
                'label': 0
            })
    return lista_dados

# --- Loop Principal para iterar de chb01 a chb24 ---
diretorio_base = DATASET_DIR
todos_os_labels = []

for i in range(1, 25):
    pasta_nome = f'chb{i:02d}'
    caminho_pasta = os.path.join(diretorio_base, pasta_nome)
    arquivo_sumario = os.path.join(caminho_pasta, f'{pasta_nome}-summary.txt')

    if os.path.exists(arquivo_sumario):
        print(f"Processando sumário da pasta: {pasta_nome}")
        dados_paciente = extrair_dados_sumario(arquivo_sumario)

        df_paciente = pd.DataFrame(dados_paciente)
        df_paciente.to_csv(os.path.join(caminho_pasta, f'{pasta_nome}_labels.csv'), index=False)

        for item in dados_paciente:
            item['patient'] = pasta_nome
            todos_os_labels.append(item)

if not todos_os_labels:
    print(f"[ERRO] Nenhum '-summary.txt' encontrado em {diretorio_base}. "
          f"Verifique se o dataset CHB-MIT está nesse caminho.")
    sys.exit(1)

# Salvar CSV Global (Muito útil para treinar o classificador com todos os dados)
df_global = pd.DataFrame(todos_os_labels)
global_labels_path = os.path.join(REPO_ROOT, 'chb_mit_global_labels.csv')
df_global.to_csv(global_labels_path, index=False)
print(f"\nProcessamento concluído! CSV Global gerado em: {global_labels_path}")
