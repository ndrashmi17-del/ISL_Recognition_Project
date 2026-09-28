import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = os.path.join("models", "sign_model.keras")
NORMALIZATION_PATH = os.path.join("models", "normalization.npz")
DATASET_PATH = "dataset"

FRAMES = 30
FEATURES = 255

ACTIONS = [
    "Hello", "Thank You", "Help", "Yes", "No",
    "Please", "Sorry", "Good Morning", "Good Night", "How Are You",
    "Welcome", "Come", "Go", "Stop", "Eat",
    "Drink", "Water", "Home", "Friend", "Love"
]

# ============================================================
# LOAD MODEL + NORMALIZATION
# ============================================================
print("\nLoading model and normalization...")
model = load_model(MODEL_PATH)
norm = np.load(NORMALIZATION_PATH)
MEAN = norm["mean"].astype(np.float32)
STD = norm["std"].astype(np.float32)
print("Loaded successfully.\n")

# ============================================================
# LOAD ALL DATA
# ============================================================
X = []
y = []

print("Loading dataset...")
for label_idx, action in enumerate(ACTIONS):
    action_path = os.path.join(DATASET_PATH, action)
    if not os.path.exists(action_path):
        print(f"Warning: Folder not found → {action}")
        continue

    files = sorted([f for f in os.listdir(action_path) if f.endswith(".npy")])
    for file in files:
        data = np.load(os.path.join(action_path, file))
        if data.shape == (FRAMES, FEATURES):
            X.append(data)
            y.append(label_idx)
        else:
            print(f"Skipped bad shape: {file} → {data.shape}")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"Total sequences loaded: {len(X)}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}\n")

# ============================================================
# NORMALIZE
# ============================================================
X_norm = (X - MEAN) / STD

# ============================================================
# PREDICT
# ============================================================
print("Running predictions...")
preds = model.predict(X_norm, verbose=0)
y_pred = np.argmax(preds, axis=1)

# ============================================================
# RESULTS
# ============================================================
correct = np.sum(y_pred == y)
total = len(y)
accuracy = correct / total

print("=" * 60)
print("RESULT")
print("=" * 60)
print(f"Correct : {correct}")
print(f"Wrong   : {total - correct}")
print(f"Total   : {total}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)

print("\nClassification Report:\n")
print(classification_report(y, y_pred, target_names=ACTIONS, digits=3))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ACTIONS, yticklabels=ACTIONS)
plt.title("Confusion Matrix - SignSpeak", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved as → confusion_matrix.png")
plt.show()