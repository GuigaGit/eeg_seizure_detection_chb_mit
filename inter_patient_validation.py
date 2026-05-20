import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score
import gc

def run_inter_patient_validation():
    print(f"{'='*50}")
    print("Iniciando Validação Inter-Pacientes (Cross-Patient)")
    print(f"{'='*50}")

    # 1. Identificar quais pacientes possuem modelos treinados
    trained_patients = []
    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        model_path = f"models/svm_linear_{patient_id}.pkl"
        if os.path.exists(model_path):
            trained_patients.append(patient_id)

    n_patients = len(trained_patients)
    if n_patients == 0:
        print("[ERRO] Nenhum modelo SVM encontrado na pasta 'models/'. Treine os modelos primeiro.")
        return

    print(f"Modelos encontrados: {n_patients} pacientes.\n")

    # Matrizes para armazenar os resultados (Linhas: Modelo Treinado | Colunas: Dados de Teste)
    matrix_sen = np.zeros((n_patients, n_patients))
    matrix_spe = np.zeros((n_patients, n_patients))

    # 2. Loop Duplo: Treino vs Teste
    for i, train_id in enumerate(trained_patients):
        print(f"Avaliando o modelo do paciente: {train_id}")
        
        # Carrega o modelo e o scaler específicos deste paciente
        model = joblib.load(f"models/svm_linear_{train_id}.pkl")
        scaler = joblib.load(f"models/scaler_{train_id}.pkl")

        for j, test_id in enumerate(trained_patients):
            # Carrega os dados do paciente de teste
            x_path = f"X_{test_id}.npy"
            y_path = f"y_{test_id}.npy"

            if not os.path.exists(x_path) or not os.path.exists(y_path):
                continue
            
            X_test_raw = np.load(x_path)
            y_test = np.load(y_path)

            # ATENÇÃO: O scaler do treino é aplicado aos dados de teste
            X_test_scaled = scaler.transform(X_test_raw)

            try:
                # Predição
                y_pred = model.predict(X_test_scaled)
            except ValueError as e:
                print(f"[ERRO DIMENSIONAL] Falha ao testar {train_id} em {test_id}: {e}")
                matrix_sen[i, j] = np.nan
                matrix_spe[i, j] = np.nan
                continue # Pula para o próximo paciente de teste sem parar o script

            # Se não houver crises nos dados, o recall_score retorna aviso. Protegemos isso:
            if sum(y_test) > 0:
                sen = recall_score(y_test, y_pred, pos_label=1)
            else:
                sen = np.nan # Not a Number, se o paciente não tiver crise
            
            if len(y_test) - sum(y_test) > 0:
                spe = recall_score(y_test, y_pred, pos_label=0)
            else:
                spe = np.nan

            # Armazena nas matrizes
            matrix_sen[i, j] = sen
            matrix_spe[i, j] = spe
            
            # Limpa as variáveis pesadas para liberar RAM
            del X_test_raw
            del X_test_scaled
            gc.collect() # Força o Python a esvaziar o lixo da memória

    # 3. Geração dos Mapas de Calor (Heatmaps)
    print("\nGerando Heatmaps de Validação Cruzada...")
    os.makedirs('results/cross_patient', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Heatmap de Sensibilidade
    sns.heatmap(matrix_sen, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=trained_patients, yticklabels=trained_patients, ax=ax1, vmin=0, vmax=1)
    ax1.set_title("Sensibilidade Inter-Paciente")
    ax1.set_xlabel("Testado no Paciente (Dados)")
    ax1.set_ylabel("Treinado no Paciente (Modelo)")

    # Heatmap de Especificidade
    sns.heatmap(matrix_spe, annot=True, fmt=".2f", cmap="Oranges", 
                xticklabels=trained_patients, yticklabels=trained_patients, ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Especificidade Inter-Paciente")
    ax2.set_xlabel("Testado no Paciente (Dados)")
    ax2.set_ylabel("Treinado no Paciente (Modelo)")

    plt.tight_layout()
    plt.savefig('results/cross_patient/inter_patient_heatmaps.png', dpi=300)
    print("Gráficos salvos em: 'results/cross_patient/inter_patient_heatmaps.png'")

if __name__ == "__main__":
    run_inter_patient_validation()