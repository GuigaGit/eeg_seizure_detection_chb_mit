import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_patient_specific_models(training_rate=0.50):
    print(f"Iniciando Treinamento Específico por Paciente (Taxa de Treino: {training_rate*100}%)")
    print("Modelo: SVM com Kernel Linear (Baseline de Comparação)")
    
    all_accuracies = []
    all_sensitivities = []
    all_specificities = []
    trained_patients = []

    # Cria pasta para salvar os gráficos das curvas
    os.makedirs('results/curves', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Iterando apenas o chb01 para teste rápido (mude para range(1, 25) quando quiser rodar todos)
    for i in [1]:
        patient_id = f"chb{i:02d}"
        x_path = f"X_{patient_id}.npy"
        y_path = f"y_{patient_id}.npy"

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue
            
        print(f"\n{'-'*40}")
        print(f"Processando Paciente: {patient_id}")
        
        X = np.load(x_path)
        y = np.load(y_path)
        
        # 1. Divisão Cronológica (Sem embaralhamento)
        split_idx = int(len(X) * training_rate)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if sum(y_train) == 0 or sum(y_test) == 0:
            print(f"[AVISO] {patient_id} sem crises no treino ou teste. Pulando paciente.")
            continue

        # 2. Padronização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Stratified Cross Validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # 4. Parâmetros do SVM Linear
        # O paper cita SVM linear. Portanto, testaremos apenas o parâmetro de regularização 'C'
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear']
        }

        print(f"[{patient_id}] Tunando hiperparâmetros (C)...")
        grid = GridSearchCV(
            SVC(class_weight='balanced'), 
            param_grid, 
            refit=True, 
            cv=skf, 
            n_jobs=-1,
            scoring='recall', # Otimizando para não perder as crises
            return_train_score=True # Necessário para plotar as curvas de treino/validação
        )
        
        grid.fit(X_train_scaled, y_train)
        
        # 5. Avaliação no conjunto de teste (futuro cronológico)
        y_pred = grid.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        sen = recall_score(y_test, y_pred, pos_label=1)
        spe = recall_score(y_test, y_pred, pos_label=0) 
        
        all_accuracies.append(acc)
        all_sensitivities.append(sen)
        all_specificities.append(spe)
        trained_patients.append(patient_id)

        print(f"[{patient_id}] Melhor Parâmetro: {grid.best_params_}")
        print(f"[{patient_id}] Sensibilidade: {sen:.4f} | Especificidade: {spe:.4f} | Acurácia: {acc:.4f}")

        # Salva os modelos
        joblib.dump(grid.best_estimator_, f'models/svm_linear_{patient_id}.pkl')
        joblib.dump(scaler, f'models/scaler_{patient_id}.pkl')

        # 6. PLOT DA CURVA DE VALIDAÇÃO DO PARÂMETRO 'C'
        results = grid.cv_results_
        c_values = np.array([params['C'] for params in results['params']])
        mean_test_scores = results['mean_test_score']
        mean_train_scores = results['mean_train_score']

        plt.figure(figsize=(10, 6))
        plt.title(f'Curva de Validação SVM Linear - Paciente {patient_id}')
        plt.xlabel('Parâmetro C (Regularização)')
        plt.ylabel('Score (Recall/Sensibilidade)')
        
        # Usamos escala logarítmica no eixo X porque os valores de C crescem exponencialmente
        plt.semilogx(c_values, mean_train_scores, label='Score de Treino', color='blue', marker='o')
        plt.semilogx(c_values, mean_test_scores, label='Score de Validação (CV)', color='orange', marker='s')
        
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Salva a curva específica do paciente
        plt.savefig(f'results/curves/validation_curve_{patient_id}.png', dpi=300)
        plt.close() # Fecha a figura para economizar memória

    # 7. Relatório Final e Gráfico Geral
    if all_accuracies:
        print(f"\n{'='*40}")
        print("PERFORMANCE GERAL DO SVM LINEAR")
        print(f"{'='*40}")
        print(f"Média Sensibilidade: {np.mean(all_sensitivities):.4f}")
        print(f"Média Especificidade: {np.mean(all_specificities):.4f}")
        print(f"Média Acurácia:      {np.mean(all_accuracies):.4f}")
        print(f"{'='*40}")

        print("\nGerando gráficos de performance geral...")
        
        patient_labels = trained_patients
        x = np.arange(len(patient_labels))
        width = 0.25

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Subplot 1: Barras por Paciente
        ax1.bar(x - width, all_sensitivities, width, label='Sensibilidade', color='#1f77b4')
        ax1.bar(x, all_specificities, width, label='Especificidade', color='#ff7f0e')
        ax1.bar(x + width, all_accuracies, width, label='Acurácia', color='#2ca02c')

        ax1.set_ylabel('Score')
        ax1.set_title('Performance SVM Linear por Paciente')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patient_labels, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim([0, 1.05])

        # Subplot 2: Boxplot para variância
        data = [all_sensitivities, all_specificities, all_accuracies]
        ax2.boxplot(data, labels=['Sensibilidade', 'Especificidade', 'Acurácia'], patch_artist=True)
        ax2.set_title('Distribuição Geral de Performance')
        ax2.set_ylabel('Score')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        
        plt.savefig('results/svm_linear_performance.png', dpi=300)
        print("Gráficos salvos na pasta 'results/'")

if __name__ == "__main__":
    train_patient_specific_models(training_rate=0.50)