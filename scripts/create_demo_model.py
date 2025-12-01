"""
Create a minimal CNN model for face recognition demo.
Model akan memprediksi identitas mahasiswa berdasarkan cropped face image.
"""
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load class indices
models_dir = Path(__file__).resolve().parents[1] / "models"
classes_path = models_dir / "class_indices.json"

with open(classes_path, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# Reverse mapping: name -> index
name_to_idx = {k: v for k, v in class_indices.items()}
num_classes = len(name_to_idx)

print(f"Total classes: {num_classes}")

# Build minimal CNN model
input_shape = (160, 160, 3)

inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

print("Model summary:")
model.summary()

# Create dummy training data (random images)
np.random.seed(42)
X_dummy = np.random.randint(0, 256, size=(num_classes * 2, 160, 160, 3), dtype="uint8").astype("float32") / 255.0
y_dummy = np.eye(num_classes)
y_dummy = np.repeat(y_dummy, 2, axis=0)

print(f"Training on dummy data: X shape {X_dummy.shape}, y shape {y_dummy.shape}")

# Train briefly on dummy data (so model weights are initialized and realistic)
model.fit(X_dummy, y_dummy, epochs=2, batch_size=8, verbose=1)

# Save model
output_file = models_dir / "recognizer.h5"
model.save(str(output_file))
print(f"✓ Model saved to {output_file}")
