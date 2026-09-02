import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from poincare_features import time_delay_embedding, get_poincare_intersections, extract_features

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Hyperparameters (full control, vs. sklearn defaults previously)
# ==========================================
LAYER1_EPOCHS = 200      # per-channel classifier (replaces LDA)
LAYER1_LR = 0.05
LAYER2_EPOCHS = 300      # combiner classifier (replaces linear SVC)
LAYER2_LR = 0.05
LAYER2_C = 1.0           # inverse regularization strength, like sklearn's SVC(C=...)


class LinearClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_logistic(model, X, y, epochs, lr):
    """Replaces sklearn's LDA with a logistic-regression linear classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_float = y.float()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X), y_float)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_linear_svm(model, X, y, C, epochs, lr):
    """Replaces sklearn's SVC(kernel='linear') with a hinge-loss linear layer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_signed = y.float() * 2 - 1
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        hinge = torch.clamp(1 - y_signed * out, min=0).mean()
        w = model.linear.weight
        loss = 0.5 * (w ** 2).sum() + C * hinge
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_labels(model, X):
    with torch.no_grad():
        return (model(X) > 0).long()


# ==========================================
# Main Pipeline & Classification
# ==========================================
def run_pipeline():
    n_channels = 23
    fs = 256  # Hz
    epoch_length = 1  # second
    n_samples = fs * epoch_length

    n_epochs = 100
    y_train = torch.tensor([1] * 50 + [0] * 50, dtype=torch.long, device=DEVICE)

    print("Extracting features for Layer 1...")
    X_train_all_channels = torch.zeros((n_epochs, n_channels, 7), device=DEVICE)

    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Replace this with your actual MNE epoch data
            raw_signal = np.random.randn(n_samples)

            embedded = time_delay_embedding(raw_signal, d=5, tau=6)
            intersections = get_poincare_intersections(embedded)
            features = extract_features(intersections)
            X_train_all_channels[epoch, ch, :] = features

    # Layer 1: Train 23 separate per-channel classifiers (LDA replacement)
    print("Training Layer 1 (23 logistic classifiers, replacing LDA)...")
    layer1_models = []
    layer_1_predictions = torch.zeros((n_epochs, n_channels), device=DEVICE)

    for ch in range(n_channels):
        model = LinearClassifier(7).to(DEVICE)
        X_ch = X_train_all_channels[:, ch, :]
        model = train_logistic(model, X_ch, y_train, epochs=LAYER1_EPOCHS, lr=LAYER1_LR)
        layer1_models.append(model)

        layer_1_predictions[:, ch] = predict_labels(model, X_ch).float()

    # Layer 2: Train linear SVM (hinge loss) on the binary outputs of Layer 1
    print("Training Layer 2 (linear SVM, PyTorch hinge loss)...")
    svm_classifier = LinearClassifier(n_channels).to(DEVICE)
    svm_classifier = train_linear_svm(
        svm_classifier, layer_1_predictions, y_train, C=LAYER2_C,
        epochs=LAYER2_EPOCHS, lr=LAYER2_LR
    )

    print("Pipeline trained successfully!")

    # Example Inference (Test on a new epoch)
    print("\n--- Running Inference on a new 1-second epoch ---")
    test_layer1_preds = torch.zeros(n_channels, device=DEVICE)

    for ch in range(n_channels):
        new_signal = np.random.randn(n_samples)
        embedded = time_delay_embedding(new_signal, d=5, tau=6)
        intersections = get_poincare_intersections(embedded)
        features = extract_features(intersections).unsqueeze(0)

        test_layer1_preds[ch] = predict_labels(layer1_models[ch], features)[0].float()

    final_prediction = predict_labels(svm_classifier, test_layer1_preds.unsqueeze(0))

    result = "Seizure Detected!" if final_prediction.item() == 1 else "Normal (Non-seizure)"
    print(f"Final System Output: {result}")


if __name__ == "__main__":
    run_pipeline()
