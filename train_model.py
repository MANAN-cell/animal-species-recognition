
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
BASE_PATH = "Animal_Dataset/Animal"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("Classes:", train_ds.class_names)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build classifier
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1. / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(train_ds.class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model only when this script is executed
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save trained model
os.makedirs("models", exist_ok=True)
model.save("models/animal_species_model.keras")
print("Model saved to models/animal_species_model.keras")

# Save class order for future predictions
with open("models/class_names.txt", "w") as f:
    f.write("\n".join(train_ds.class_names))
print("Class names saved.")
