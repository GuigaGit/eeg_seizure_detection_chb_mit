import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Carregar os dados gerados pelo label_dataset_v2.py
print("Carregando dados...")
X = np.load('X_final.npy')
y = np.load('y_final.npy')

# 2. Divisão de Treino e Teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Normalização (CRUCIAL para SVM)
# O SVM calcula distâncias. Se uma característica (como variância) tiver escala 
# muito diferente de outra (como frequência), o modelo será enviesado.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Treinamento do Modelo SVM
print("Treinando o SVM...")
# Usamos kernel='rbf' (padrão) e class_weight='balanced' para lidar com 
# o desequilíbrio entre janelas de crise e não-crise.
model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 5. Avaliação
y_pred = model.predict(X_test_scaled)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Crise']))

# 6. Matriz de Confusão Visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Crise'], yticklabels=['Normal', 'Crise'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Detecção de Crises')
plt.show()

# 7. Salvar o modelo e o scaler para uso futuro
joblib.dump(model, 'svm_seizure_model.pkl')
joblib.dump(scaler, 'scaler_svm.pkl')
print("\nModelo e Scaler salvos com sucesso!")