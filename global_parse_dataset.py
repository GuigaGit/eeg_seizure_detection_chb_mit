import pandas as pd
import re
import os

def extrair_dados_sumario(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        conteudo = f.read()

    blocos = conteudo.split('File Name: ')
    lista_dados = []

    for bloco in blocos[1:]:
        linhas = bloco.split('\n')
        nome_arquivo = linhas[0].strip()
        
        # Regex mais robusto para capturar números mesmo com espaços extras
        match_crises = re.search(r'Number of Seizures in File:\s*(\d+)', bloco)
        num_crises = int(match_crises.group(1)) if match_crises else 0
        
        if num_crises > 0:
            # Captura tempos garantindo que ignore variações de texto entre "Seizure 1" e "Seizure Start"
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
diretorio_base = './dataset_chbmit' # Ajuste para o seu caminho
todos_os_labels = []

for i in range(1, 25):
    pasta_nome = f'chb{i:02d}'
    caminho_pasta = os.path.join(diretorio_base, pasta_nome)
    arquivo_sumario = os.path.join(caminho_pasta, f'{pasta_nome}-summary.txt')
    
    if os.path.exists(arquivo_sumario):
        print(f"Processando sumário da pasta: {pasta_nome}")
        dados_paciente = extrair_dados_sumario(arquivo_sumario)
        
        # Opcional: Salvar um CSV por paciente
        df_paciente = pd.DataFrame(dados_paciente)
        df_paciente.to_csv(os.path.join(caminho_pasta, f'{pasta_nome}_labels.csv'), index=False)
        
        # Acumular para um CSV global
        for item in dados_paciente:
            item['patient'] = pasta_nome
            todos_os_labels.append(item)

# Salvar CSV Global (Muito útil para treinar o SVM com todos os dados)
df_global = pd.DataFrame(todos_os_labels)
df_global.to_csv('chb_mit_global_labels.csv', index=False)
print("\nProcessamento concluído! CSV Global gerado.")