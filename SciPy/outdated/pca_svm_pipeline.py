import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import warnings

# Suppress warnings for clean output during demonstration
warnings.filterwarnings('ignore')

# ==========================================
# 1. Phase Space Reconstruction (PSR)
# ==========================================
def time_delay_embedding(signal, d=5, tau=6):
    """
    Reconstructs the phase space using time-delay embedding.
    d: embedding dimension (5 per the paper)
    tau: time lag (6 per the paper, ~23ms at 256Hz)
    """
    n_samples = len(signal)
    valid_length = n_samples - (d - 1) * tau
    
    if valid_length <= 0:
        raise ValueError("Signal is too short for the chosen d and tau.")
        
    embedded = np.zeros((valid_length, d))
    for i in range(d):
        embedded[:, i] = signal[i*tau : valid_length + i*tau]
        
    return embedded

# ==========================================
# 2. PCA & Poincaré Section Mapping
# ==========================================
def get_poincare_intersections(embedded_space):
    """
    Applies PCA, fits a 1st-degree polynomial (line), and finds intersections.
    Returns the PC1 values of the intersection points.
    """
    # Apply PCA to reduce 5D to 2D
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embedded_space)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    
    # Fit a 1st-degree polynomial (line) to the 2D space: pc2 = m * pc1 + c
    m, c = np.polyfit(pc1, pc2, 1)
    
    intersection_pc1_values = []
    
    # Find intersections of the trajectory with the fitted line
    # A trajectory segment goes from point i to i+1.
    # The line equation is f(x,y) = y - mx - c = 0
    for i in range(len(pc1) - 1):
        x1, y1 = pc1[i], pc2[i]
        x2, y2 = pc1[i+1], pc2[i+1]
        
        f1 = y1 - (m * x1 + c)
        f2 = y2 - (m * x2 + c)
        
        # If the signs are opposite, the trajectory crossed the line
        if f1 * f2 < 0:
            # Linear interpolation to find the exact x (PC1) intersection point
            # 0 = (y - y1) - m*(x - x1) -> solving for intersection
            # To avoid complex line geometry math, we interpolate based on the fraction of the crossing
            fraction = abs(f1) / (abs(f1) + abs(f2))
            intersect_x = x1 + fraction * (x2 - x1)
            intersection_pc1_values.append(intersect_x)
            
    return np.array(intersection_pc1_values)

# ==========================================
# 3. Feature Extraction
# ==========================================
def extract_features(intersections):
    """
    Extracts the 7 statistical features from the intersection points.
    """
    # If no intersections exist, return zeros to avoid NaN errors
    if len(intersections) < 2:
        return np.zeros(7)
        
    # 1. Range
    rng = np.max(intersections) - np.min(intersections)
    
    # 2. 0.13 Quantile
    q_013 = np.quantile(intersections, 0.13)
    
    # 3. Interquartile Range (IQR)
    iqr = np.percentile(intersections, 75) - np.percentile(intersections, 25)
    
    # 4. Shannon Entropy
    # We estimate the PDF using a histogram
    hist, bin_edges = np.histogram(intersections, density=True, bins='auto')
    hist = hist[hist > 0] # Remove zeros for log2
    entropy = -np.sum(hist * np.log2(hist))
    
    # 5. Root Mean Squared Amplitude
    rms = np.sqrt(np.mean(intersections**2))
    
    # 6. Coefficient of Variation
    mean_val = np.mean(intersections)
    cov = np.std(intersections) / mean_val if mean_val != 0 else 0
    
    # 7. Energy
    energy = np.sum(intersections**2)
    
    return np.array([rng, q_013, iqr, entropy, rms, cov, energy])

# ==========================================
# 4. Main Pipeline & Classification
# ==========================================
# Layer 2 modes:
#   'svc'   -> supervised "seizure vs non-seizure" classifier (original paper setup)
#   'ocsvm' -> "anomaly vs normal" detector: OneClassSVM trained only on non-seizure epochs
MODES = ['svc', 'ocsvm']

def simulate_epoch_features(n_epochs, n_channels, n_samples):
    """
    Runs PSR -> Poincare -> features for every epoch/channel.
    Returns shape (epochs, channels, 7).
    """
    X = np.zeros((n_epochs, n_channels, 7))
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Replace this with your actual MNE epoch data
            raw_signal = np.random.randn(n_samples)
            embedded = time_delay_embedding(raw_signal, d=5, tau=6)
            intersections = get_poincare_intersections(embedded)
            X[epoch, ch, :] = extract_features(intersections)
    return X

def layer1_outputs(lda_models, X_all_channels, mode):
    """
    Builds the Layer 2 input from the 23 LDA classifiers.
    'svc'  : binary LDA votes (as in the paper).
    'ocsvm': continuous LDA scores (signed distance to the LDA boundary).
             The OneClassSVM only sees non-seizure epochs, whose binary votes are
             almost all 0 -> a near-degenerate training set. The scores keep the
             "how confident" information the anomaly detector needs.
    """
    n_epochs, n_channels, _ = X_all_channels.shape
    out = np.zeros((n_epochs, n_channels))
    for ch, lda in enumerate(lda_models):
        X_ch = X_all_channels[:, ch, :]
        out[:, ch] = lda.predict(X_ch) if mode == 'svc' else lda.decision_function(X_ch)
    return out

def train_layer2(layer_1_outputs, y_train, mode):
    if mode == 'svc':
        # Future test: change kernel to 'rbf' or 'poly' if needed, but 'linear' is a good start for binary classification
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(layer_1_outputs, y_train)
    else:
        # nu: upper bound on the fraction of normal training epochs treated as outliers,
        # i.e. roughly the false-alarm rate you accept on normal EEG.
        clf = make_pipeline(StandardScaler(), OneClassSVM(kernel='rbf', gamma='scale', nu=0.05))
        clf.fit(layer_1_outputs[y_train == 0])  # Normal (non-seizure) epochs only
    return clf

def predict_layer2(clf, layer_1_outputs, mode):
    """Returns 1 = Seizure, 0 = Non-seizure for both modes."""
    pred = clf.predict(layer_1_outputs)
    if mode == 'ocsvm':
        # OneClassSVM: +1 = inlier (normal), -1 = outlier (anomaly -> seizure)
        pred = (pred == -1).astype(int)
    return pred

def run_pipeline(mode='svc', seed=42):
    # Same seed for every mode so they are compared on identical data
    np.random.seed(seed)

    # Simulation Parameters
    n_channels = 23
    fs = 256 # Hz
    epoch_length = 1 # second
    n_samples = fs * epoch_length

    # Generate synthetic data (e.g., 50 epochs of seizure, 50 of non-seizure)
    y_train = np.array([1]*50 + [0]*50) # 1 = Seizure, 0 = Non-seizure
    y_test = np.array([1]*20 + [0]*20)

    print(f"\n===== Mode: {mode} =====")
    print("Extracting features for Layer 1...")
    X_train_all_channels = simulate_epoch_features(len(y_train), n_channels, n_samples)
    X_test_all_channels = simulate_epoch_features(len(y_test), n_channels, n_samples)

    # Layer 1: Train 23 separate LDA classifiers (supervised in both modes)
    print("Training Layer 1 (23 LDA classifiers)...")
    lda_models = []
    for ch in range(n_channels):
        lda = LDA()
        lda.fit(X_train_all_channels[:, ch, :], y_train)
        lda_models.append(lda)

    # Layer 2
    print(f"Training Layer 2 ({'SVC' if mode == 'svc' else 'OneClassSVM'})...")
    train_l1 = layer1_outputs(lda_models, X_train_all_channels, mode)
    layer2 = train_layer2(train_l1, y_train, mode)
    print("Pipeline trained successfully!")

    # Evaluation on held-out epochs
    test_l1 = layer1_outputs(lda_models, X_test_all_channels, mode)
    y_pred = predict_layer2(layer2, test_l1, mode)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,  # seizures caught
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,  # normal correctly ignored
        'accuracy': (tp + tn) / len(y_test),
    }
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print("  ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
    return metrics

if __name__ == "__main__":
    results = {mode: run_pipeline(mode) for mode in MODES}

    print("\n===== Comparison =====")
    print(f"{'mode':<8}{'sensitivity':>13}{'specificity':>13}{'accuracy':>10}")
    for mode, m in results.items():
        print(f"{mode:<8}{m['sensitivity']:>13.2f}{m['specificity']:>13.2f}{m['accuracy']:>10.2f}")
