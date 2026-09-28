import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.keras")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "normalization.npz")

FRAMES = 30
FEATURES = 255

ACTIONS = [
    "Hello",
    "Thank You",
    "Help",
    "Yes",
    "No",
    "Please",
    "Sorry",
    "Good Morning",
    "Good Night",
    "How Are You",
    "Welcome",
    "Come",
    "Go",
    "Stop",
    "Eat",
    "Drink",
    "Water",
    "Home",
    "Friend",
    "Love"
]

NUM_CLASSES = len(ACTIONS)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("SIGN-SPEAK MODEL TRAINING")
print("========================================\n")

X = []
y = []

print("Loading dataset...\n")

for label, action in enumerate(ACTIONS):

    action_path = os.path.join(DATASET_DIR, action)

    if not os.path.isdir(action_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {action_path}"
        )

    files = sorted(
        [
            file for file in os.listdir(action_path)
            if file.lower().endswith(".npy")
        ]
    )

    print(f"{action:15s}: {len(files)} sequences")

    if len(files) == 0:
        raise ValueError(
            f"No .npy files found for sign: {action}"
        )

    for file in files:

        file_path = os.path.join(action_path, file)

        sequence = np.load(file_path)

        if sequence.shape != (FRAMES, FEATURES):
            raise ValueError(
                f"\nWrong shape in:\n{file_path}\n"
                f"Expected: {(FRAMES, FEATURES)}\n"
                f"Found:    {sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                f"\nNaN or infinity found in:\n{file_path}"
            )

        X.append(sequence.astype(np.float32))
        y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\n========================================")
print("DATASET SUMMARY")
print("========================================")
print("Total sequences :", len(X))
print("Input shape     :", X.shape)
print("Number of signs :", NUM_CLASSES)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nCalculating normalization values...")

mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True)

std = np.where(std < 1e-6, 1.0, std)

# Normalize training data
X = (X - mean) / std

X = X.astype(np.float32)

print("Normalization completed.")


# ============================================================
# SAVE NORMALIZATION VALUES
# ============================================================

np.savez(
    NORMALIZATION_PATH,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32)
)

print(f"Normalization saved to:")
print(NORMALIZATION_PATH)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("DATA SPLIT")
print("========================================")
print("Training samples   :", len(X_train))
print("Validation samples :", len(X_val))


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_train_cat = to_categorical(
    y_train,
    num_classes=NUM_CLASSES
)

y_val_cat = to_categorical(
    y_val,
    num_classes=NUM_CLASSES
)


# ============================================================
# MODEL
# ============================================================

print("\n========================================")
print("BUILDING MODEL")
print("========================================\n")

model = Sequential([
    LSTM(
        128,
        return_sequences=True,
        activation="tanh",
        input_shape=(FRAMES, FEATURES)
    ),

    BatchNormalization(),
    Dropout(0.30),

    LSTM(
        128,
        return_sequences=True,
        activation="tanh"
    ),

    BatchNormalization(),
    Dropout(0.30),

    LSTM(
        64,
        return_sequences=False,
        activation="tanh"
    ),

    Dropout(0.30),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(64, activation="relu"),

    Dense(NUM_CLASSES, activation="softmax")
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=20,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=150,
    batch_size=16,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

best_model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# FINAL VALIDATION
# ============================================================

loss, accuracy = best_model.evaluate(
    X_val,
    y_val_cat,
    verbose=0
)

print("\n========================================")
print("FINAL RESULT")
print("========================================")

print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation loss    : {loss:.4f}")

print("\n========================================")
print("FILES SAVED")
print("========================================")

print(f"Model          : {MODEL_PATH}")
print(f"Normalization  : {NORMALIZATION_PATH}")

print("\nTraining completed successfully!")