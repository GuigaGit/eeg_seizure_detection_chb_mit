import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def train_patient_specific_models(training_rate=0.50):
    """
    Trains an independent SVM model for each patient using a chronological split,
    replicating the methodology of Zabihi et al.
    """
    print(f"Starting Patient-Specific Training (Training Rate: {training_rate*100}%)")
    
    # Storage for overall metrics to calculate averages at the end
    all_accuracies = []
    all_sensitivities = []
    all_specificities = []

    # Iterate through all 24 patients in the CHB-MIT dataset
    for i in range(1, 25):
        patient_id = f"chb{i:02d}"
        x_path = f"X_{patient_id}.npy"
        y_path = f"y_{patient_id}.npy"

        # Skip if the patient's data hasn't been generated
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue
            
        print(f"\n{'-'*40}")
        print(f"Processing Patient: {patient_id}")
        
        X = np.load(x_path)
        y = np.load(y_path)
        
        # 1. Chronological Split (NO random shuffling)
        split_idx = int(len(X) * training_rate)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Ensure there are actually seizures in both the train and test sets for evaluation
        if sum(y_train) == 0 or sum(y_test) == 0:
            print(f"[WARNING] {patient_id} lacks seizure data in either train or test split. Skipping model evaluation.")
            continue

        # 2. Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Time-Series Cross Validation for GridSearch
        # This prevents "future" epochs from leaking into validation sets during hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=3)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.01, 0.001],
            'kernel': ['rbf']
        }

        print(f"[{patient_id}] Tuning hyperparameters...")
        grid = GridSearchCV(
            SVC(class_weight='balanced'), 
            param_grid, 
            refit=True, 
            cv=tscv, 
            n_jobs=-1,
            scoring='recall' # Optimizing for sensitivity (detecting seizures)
        )
        
        grid.fit(X_train_scaled, y_train)
        
        # 4. Evaluation on the strictly unseen chronological test set
        y_pred = grid.predict(X_test_scaled)
        
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        # Sensitivity (Recall for class 1)
        sen = recall_score(y_test, y_pred, pos_label=1)
        # Specificity (Recall for class 0)
        spe = recall_score(y_test, y_pred, pos_label=0) 
        
        all_accuracies.append(acc)
        all_sensitivities.append(sen)
        all_specificities.append(spe)

        print(f"[{patient_id}] Best Params: {grid.best_params_}")
        print(f"[{patient_id}] Sensitivity: {sen:.4f} | Specificity: {spe:.4f} | Accuracy: {acc:.4f}")

        # 5. Save the patient-specific model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(grid.best_estimator_, f'models/svm_{patient_id}.pkl')
        joblib.dump(scaler, f'models/scaler_{patient_id}.pkl')

    # 6. Final Aggregate Report
    if all_accuracies:
        print(f"\n{'='*40}")
        print("OVERALL PIPELINE PERFORMANCE")
        print(f"{'='*40}")
        print(f"Average Sensitivity: {np.mean(all_sensitivities):.4f}")
        print(f"Average Specificity: {np.mean(all_specificities):.4f}")
        print(f"Average Accuracy:    {np.mean(all_accuracies):.4f}")
        print(f"{'='*40}")

if __name__ == "__main__":
    # You can easily change this to 0.25 to replicate the 25% training rate experiment
    train_patient_specific_models(training_rate=0.50)