
# train_model.py — Stable, Fine-Tuned Version


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# 1. Paths and Setup

train_dir = "dataset/"
save_dir = "model"

os.makedirs(save_dir, exist_ok=True)
print(f"✅ Model directory ready: {save_dir}/")


# 2. Data Augmentation & Preprocessing

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("✅ Classes detected:", train_data.class_indices)


# 3. Model Architecture — MobileNetV2 Fine-Tuned

base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers for transfer learning
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])


# 4. Compile Model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 5. Callbacks for Smart Training

checkpoint_path = os.path.join(save_dir, "best_model.h5")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
]


# 6. Train Model

print("\n🚀 Starting model training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks
)


# 7. Evaluate Model

loss, acc = model.evaluate(val_data)
print(f"\n✅ Validation Accuracy: {acc:.3f} | Loss: {loss:.3f}")

# 8. Save Final Model (Flask Compatible)

final_model_path = os.path.join(save_dir, "pattern_model.h5")
model.save(final_model_path)
print(f"💾 Final model saved at: {final_model_path}")


# 9. Save Class Labels

class_indices = train_data.class_indices
labels_path = os.path.join(save_dir, "labels.txt")

with open(labels_path, "w") as f:
    for label, idx in class_indices.items():
        f.write(f"{idx}:{label}\n")

print(f"📄 Class labels saved at: {labels_path}")
print("✅ All done! You can now run app.py to test predictions.")
