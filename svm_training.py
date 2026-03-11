import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def train_model():
    X = np.load('X_final.npy')
    y = np.load('y_final.npy')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definição do espaço de busca para o SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf']
    }

    print("Iniciando Grid Search paralelo...")
    # n_jobs=-1 aqui distribui os diferentes modelos pelos núcleos
    grid = GridSearchCV(
        SVC(class_weight='balanced'), 
        param_grid, 
        refit=True, 
        verbose=2, 
        cv=5, 
        n_jobs=-1
    )
    
    grid.fit(X_train_scaled, y_train)

    print(f"Melhores parâmetros: {grid.best_params_}")
    
    y_pred = grid.predict(X_test_scaled)
    print("\n--- Relatório Final ---")
    print(classification_report(y_test, y_pred))

    # Salvar o melhor modelo
    joblib.dump(grid.best_estimator_, 'svm_seizure_model.pkl')
    joblib.dump(scaler, 'scaler_svm.pkl')

if __name__ == "__main__":
    train_model()