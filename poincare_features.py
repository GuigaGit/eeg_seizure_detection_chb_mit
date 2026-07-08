import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from joblib import Parallel, delayed, cpu_count
import warnings
import mne  # <-- ADDED: MNE library for EEG processing

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR') # Keeps MNE from printing excessive loading logs

# ==========================================
# 1. Phase Space Reconstruction (PSR)
# ==========================================
def time_delay_embedding(signal, d=5, tau=6):
    """
    Reconstructs the phase space using time-delay embedding.
    """
    n_samples = len(signal)
    valid_length = n_samples - (d - 1) * tau
    
    if valid_length <= 0:
        # Fallback to prevent crashes on extremely short signal edges
        return np.zeros((1, d)) 
        
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
    if embedded_space.shape[0] < 2:
        return np.array([])
        
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embedded_space)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    
    # Protect against flat-line signals (zero variance)
    if np.var(pc1) == 0:
        return np.array([])
    
    m, c = np.polyfit(pc1, pc2, 1)
    intersection_pc1_values = []
    
    for i in range(len(pc1) - 1):
        x1, y1 = pc1[i], pc2[i]
        x2, y2 = pc1[i+1], pc2[i+1]
        
        f1 = y1 - (m * x1 + c)
        f2 = y2 - (m * x2 + c)
        
        if f1 * f2 < 0:
            fraction = abs(f1) / (abs(f1) + abs(f2) + 1e-9) # 1e-9 prevents division by zero
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
    if len(intersections) < 2:
        return np.zeros(7)
        
    rng = np.max(intersections) - np.min(intersections)
    q_013 = np.quantile(intersections, 0.13)
    iqr = np.percentile(intersections, 75) - np.percentile(intersections, 25)
    
    hist, _ = np.histogram(intersections, density=True, bins='auto')
    hist = hist[hist > 0] 
    entropy = -np.sum(hist * np.log2(hist))
    
    rms = np.sqrt(np.mean(intersections**2))
    
    mean_val = np.mean(intersections)
    cov = np.std(intersections) / mean_val if mean_val != 0 else 0
    energy = np.sum(intersections**2)
    
    return np.array([rng, q_013, iqr, entropy, rms, cov, energy])

# ==========================================
# 4. Multiprocessing Helper
# ==========================================
def compute_epoch_channel_features(epoch, ch, raw_signal_slice):
    """
    Receives an exact 1D NumPy array representing a specific epoch and channel.
    This prevents memory overhead from passing the entire dataset to every core.
    """
    embedded = time_delay_embedding(raw_signal_slice, d=5, tau=6)
    intersections = get_poincare_intersections(embedded)
    features = extract_features(intersections)
    
    return epoch, ch, features

# ==========================================
# 5. Main Pipeline
# ==========================================
def process_eeg_file(edf_file_path):
    print(f"Loading EEG data from: {edf_file_path}")
    
    # 1. Load the data using MNE
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    
    # Optional: Filter the data or pick specific channels here
    # raw.filter(l_freq=0.5, h_freq=45.0)
    # raw.pick_channels(['FP1-F7', 'F7-T7', ...])
    
    sfreq = int(raw.info['sfreq'])
    n_channels = len(raw.ch_names)
    
    # Extract the underlying NumPy array and scale it 
    # (multiplying by 1e6 converts Volts to microVolts, preventing PCA float underflow)
    data = raw.get_data() * 1e6 
    
    # 2. Windowing Parameters
    epoch_length = 1 # 1 second windows
    n_samples = sfreq * epoch_length
    
    # Calculate how many full 1-second epochs we can extract
    n_epochs = data.shape[1] // n_samples
    
    # Create an empty matrix to hold our new features
    # Shape: (epochs, channels, features)
    X_features = np.zeros((n_epochs, n_channels, 7))
    
    print(f"Data loaded: {n_channels} channels, {n_epochs} epochs. Sampling Rate: {sfreq}Hz")
    
    cores_to_use = cpu_count()
    print(f"Extracting features using {cores_to_use} cores...")
    
    # 3. Multiprocessing the feature extraction
    # We slice the data array BEFORE sending it to the helper function
    parallel_results = Parallel(n_jobs=-1)(
        delayed(compute_epoch_channel_features)(
            epoch, 
            ch, 
            data[ch, (epoch * n_samples) : ((epoch + 1) * n_samples)] # The specific signal slice
        )
        for epoch in range(n_epochs)
        for ch in range(n_channels)
    )
    
    # Reassemble the results into our 3D matrix
    for epoch, ch, features in parallel_results:
        X_features[epoch, ch, :] = features
        
    print("Feature extraction complete!")
    return X_features, raw.ch_names

if __name__ == "__main__":
    # Note: Replace 'your_eeg_record.edf' with an actual file path on your system
    edf_path = 'your_eeg_record.edf' 
    
    try:
        X_features, channel_names = process_eeg_file(edf_path)
        
        print("\n--- Summary ---")
        print(f"Final Feature Matrix Shape: {X_features.shape}")
        
        # Example of how to flatten this data for standard SVM training 
        # (Converting from 3D to 2D matrix)
        n_epochs, n_channels, n_feat = X_features.shape
        X_flattened = X_features.reshape(n_epochs, n_channels * n_feat)
        print(f"Flattened Matrix Shape for SVM: {X_flattened.shape}")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find the file '{edf_path}'. Please provide a valid MNE-compatible EEG file.")