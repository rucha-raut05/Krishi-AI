import os
import json
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset', 'PlantVillage')

MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'disease_classes.json')

# FAST SETTINGS
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 2

print("=" * 50)
print("Krishi AI - Fast Disease Training")
print("=" * 50)

# Dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_indices = train_gen.class_indices

classes_list = [None] * len(class_indices)

for cls, idx in class_indices.items():
    classes_list[idx] = cls

with open(CLASSES_PATH, 'w') as f:
    json.dump(classes_list, f, indent=2)

print("Classes saved!")

# MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(64, activation='relu')(x)

predictions = Dense(
    len(classes_list),
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining started...\n")

# Train FAST
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=5,
    validation_steps=2
)

# Save model
model.save(MODEL_PATH)

print("\nModel trained successfully!")
print(f"Saved at: {MODEL_PATH}")