import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_model():
    # 1. Carregar dados
    if not (os.path.exists('X_final.npy') and os.path.exists('y_final.npy')):
        print("Erro: Arquivos .npy não encontrados.")
        return

    X = np.load('X_final.npy')
    y = np.load('y_final.npy')
    print(f"Dados carregados. X: {X.shape}, y: {y.shape}")

    # 2. Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Treinamento
    print("Treinando SVM (Kernel RBF)...")
    model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
    model.fit(X_train_scaled, y_train)

    # 5. Avaliação
    y_pred = model.predict(X_test_scaled)
    print("\n--- Relatório de Performance ---")
    print(classification_report(y_test, y_pred))

    # 6. Salvar Artefatos
    joblib.dump(model, 'svm_seizure_model.pkl')
    joblib.dump(scaler, 'scaler_svm.pkl')
    print("Modelo e Scaler salvos com sucesso.")

if __name__ == "__main__":
    import os
    train_model()