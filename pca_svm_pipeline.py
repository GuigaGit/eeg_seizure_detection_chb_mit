import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import MultinomialNB
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
def run_pipeline():
    # Simulation Parameters
    n_channels = 23
    fs = 256 # Hz
    epoch_length = 1 # second
    n_samples = fs * epoch_length
    
    # Generate synthetic training data (e.g., 50 epochs of seizure, 50 of non-seizure)
    n_epochs = 100
    y_train = np.array([1]*50 + [0]*50) # 1 = Seizure, 0 = Non-seizure
    
    print("Extracting features for Layer 1...")
    # Shape: (epochs, channels, features)
    X_train_all_channels = np.zeros((n_epochs, n_channels, 7))
    
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Replace this with your actual MNE epoch data
            raw_signal = np.random.randn(n_samples) 
            
            # 1. PSR
            embedded = time_delay_embedding(raw_signal, d=5, tau=6)
            
            # 2. Poincare
            intersections = get_poincare_intersections(embedded)
            
            # 3. Features
            features = extract_features(intersections)
            X_train_all_channels[epoch, ch, :] = features

    # Layer 1: Train 23 separate LDA classifiers
    print("Training Layer 1 (23 LDA classifiers)...")
    lda_models = []
    layer_1_predictions = np.zeros((n_epochs, n_channels))
    
    for ch in range(n_channels):
        lda = LDA()
        # Train LDA for this specific channel
        X_ch = X_train_all_channels[:, ch, :]
        lda.fit(X_ch, y_train)
        lda_models.append(lda)
        
        # Get predictions to feed into the 2nd layer
        layer_1_predictions[:, ch] = lda.predict(X_ch)

    # Layer 2: Train Naive Bayes on the binary outputs of the LDAs
    print("Training Layer 2 (Naive Bayes)...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(layer_1_predictions, y_train)
    
    print("Pipeline trained successfully!")
    
    # Example Inference (Test on a new epoch)
    print("\n--- Running Inference on a new 1-second epoch ---")
    test_layer1_preds = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # Simulate new data
        new_signal = np.random.randn(n_samples)
        embedded = time_delay_embedding(new_signal, d=5, tau=6)
        intersections = get_poincare_intersections(embedded)
        features = extract_features(intersections).reshape(1, -1)
        
        # LDA Prediction for this channel
        test_layer1_preds[ch] = lda_models[ch].predict(features)[0]
        
    # Final Naive Bayes Prediction
    final_prediction = nb_classifier.predict(test_layer1_preds.reshape(1, -1))
    
    result = "Seizure Detected!" if final_prediction[0] == 1 else "Normal (Non-seizure)"
    print(f"Final System Output: {result}")

if __name__ == "__main__":
    run_pipeline()