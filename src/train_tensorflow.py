import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
# Paths
train_dir = "//DATASET/train"
test_dir = "//DATASET/test"

# Settings optimized for RTX 4060 8GB
img_size = 224  # Optimal for transfer learning
batch_size = 64  # Increased for RTX 4060 8GB
num_classes = 8
input_shape = (img_size, img_size, 3)
learning_rate = 0.001
epochs = 150  # More epochs with better hardware

# GPU Configuration - Fixed for proper detection
print("🔧 Configuring GPU...")
print("TensorFlow version:", tf.__version__)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {len(gpus)} GPU(s) available")

        # Print GPU details
        for i, gpu in enumerate(gpus):
            print(f"📊 GPU {i}: {gpu.name}")

        # Set GPU as default device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("✅ GPU set as default device")

    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
        print("🔄 Falling back to CPU")
else:
    print("⚠️  No GPU found, using CPU")
    print("💡 To use GPU, ensure CUDA and cuDNN are properly installed")

# Create results directory
os.makedirs("../results", exist_ok=True)

# =============================================================================
# DATA PREPROCESSING WITH ENHANCED AUGMENTATION
# =============================================================================
print("🔄 Setting up data preprocessing...")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,  # 20% for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")
print(f"✅ Class indices: {train_generator.class_indices}")


# =============================================================================
# MODEL ARCHITECTURE OPTIONS
# =============================================================================

def create_custom_cnn():
    """Create a custom CNN architecture"""
    model = Sequential([
        # First block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Second block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Third block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Fourth block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_optimized_efficientnet():
    """Create an optimized EfficientNetB3 model for RTX 4060"""
    base_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model initially
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model, base_model


def create_efficientnet_b0():
    """Create EfficientNetB0 model - lighter version"""
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model, base_model


def create_resnet_model():
    """Create a transfer learning model using ResNet50"""
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    return model, base_model


# =============================================================================
# MODEL SELECTION AND COMPILATION
# =============================================================================
print("🏗️ Creating model...")

# Choose your model (optimized for RTX 4060)
# Option 1: Custom CNN (good baseline)
# model = create_custom_cnn()
# base_model = None

# Option 2: EfficientNetB0 (Recommended - good balance of speed and accuracy)
model, base_model = create_efficientnet_b0()

# Option 3: EfficientNetB3 (Best accuracy but slower)
# model, base_model = create_optimized_efficientnet()

# Option 4: ResNet50 (Good alternative)
# model, base_model = create_resnet_model()

# Compile model
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model architecture:")
model.summary()

# =============================================================================
# CALLBACKS
# =============================================================================
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'results/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================================================
# TRAINING PHASE 1: FROZEN BASE MODEL
# =============================================================================
print("🚀 Starting training phase 1 (frozen base model)...")

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

print(f"📊 Steps per epoch: {steps_per_epoch}")
print(f"📊 Validation steps: {validation_steps}")

# Training without problematic parameters
history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=50,  # Phase 1: 50 epochs
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# TRAINING PHASE 2: FINE-TUNING (for transfer learning models)
# =============================================================================
if base_model is not None:
    print("🔥 Starting training phase 2 (fine-tuning)...")

    # Unfreeze base model
    base_model.trainable = True

    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=learning_rate / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Reset callbacks for phase 2
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'results/best_model_finetuned.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    history_phase2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=100,  # Phase 2: 100 epochs
        callbacks=callbacks,
        verbose=1
    )

    # Combine histories
    history = {
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    }
else:
    history = history_phase1.history

# =============================================================================
# SAVE FINAL MODEL
# =============================================================================
model.save("results/emotion_model_final.h5")
print("✅ Model trained and saved!")

# =============================================================================
# EVALUATION ON TEST SET
# =============================================================================
print("📊 Evaluating on test set...")

# Load best model
if base_model is not None:
    if os.path.exists('results/best_model_finetuned.h5'):
        model.load_weights('results/best_model_finetuned.h5')
    else:
        model.load_weights('results/best_model.h5')
else:
    model.load_weights('results/best_model.h5')

# Calculate test steps
test_steps = test_generator.samples // batch_size

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# Generate predictions for detailed analysis
test_generator.reset()
predictions = model.predict(test_generator, steps=test_steps, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true classes (limited to predicted samples)
true_classes = test_generator.classes[:len(predicted_classes)]

# Get class names
class_names = list(test_generator.class_indices.keys())

# Classification report
print("\n📋 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# =============================================================================
# VISUALIZATION
# =============================================================================
print("📈 Creating visualizations...")

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Confusion matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_emotion(image_path):
    """
    Predict emotion from a single image
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    emotion = class_names[predicted_class]

    return emotion, confidence


# Example usage
# emotion, confidence = predict_emotion("path/to/your/image.jpg")
# print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f})")

print("\n🎉 Training complete! Results saved in 'results' folder.")
print(f"📊 Final Test Accuracy: {test_accuracy:.4f}")
print("🔮 Use predict_emotion() function for new predictions.")

# =============================================================================
# PERFORMANCE OPTIMIZATION TIPS FOR YOUR SETUP
# =============================================================================
print("\n🚀 OPTIMIZATION TIPS FOR YOUR RTX 4060 8GB + i9-13900HX:")
print("💡 1. Batch size optimized to 64 for your 8GB VRAM")
print("💡 2. Using EfficientNetB0 for good balance of speed and accuracy")
print("💡 3. If GPU not detected, install: pip install tensorflow-gpu")
print("💡 4. For CUDA support: install CUDA 11.8 and cuDNN 8.6")
print("💡 5. Expected training time: 2-4 hours for full training")
print("💡 6. Expected accuracy: 85-95% with proper dataset")

# =============================================================================
# GPU TROUBLESHOOTING
# =============================================================================
print("\n🔧 GPU TROUBLESHOOTING:")
print("If GPU not detected, try these steps:")
print("1. pip uninstall tensorflow")
print("2. pip install tensorflow[and-cuda]")
print("3. Or install CUDA Toolkit 11.8 from NVIDIA")
print("4. Restart your system after installation")