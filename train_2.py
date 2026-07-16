import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to the MAIN folder that holds 'train' and 'test'
DATASET_PATH = r"C:\Users\Darshan\Documents\vscode\Machine Learning Project\Dataset 2\TumourClassificationImages"

TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

# ENHANCEMENT 1: High Epochs (The model stops early if it finishes learning)
EPOCHS = 50 

print("--> Initializing Training for Dataset 2 (Train/Test Structure)")
print(f"--> Training Data: {TRAIN_DIR}")
print(f"--> Testing Data:  {TEST_DIR}")

# ==========================================
# 2. DATA LOADING
# ==========================================
try:
    # 1. Load Training Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 2. Load Testing Data (Used for Validation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\n--> Classes Found: {class_names}")

except Exception as e:
    print(f"\n[CRITICAL ERROR] Could not load dataset.\n{e}")
    print("Double check that your folder contains 'train' and 'test' subfolders.")
    exit()

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization Layer
normalization_layer = layers.Rescaling(1./255)

# ENHANCEMENT 2: Data Augmentation
# This solves the "Meningioma problem" by creating fake variations
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.RandomRotation(0.2), 
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
])

# ==========================================
# 3. BUILD THE DEEP CNN MODEL (UPDATED)
# ==========================================
num_classes = len(class_names)

model = models.Sequential([
    data_augmentation,
    normalization_layer,
    
    # Conv Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Conv Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Conv Block 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # --- NEW: Conv Block 4 ---
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Dropout (Increased to 40% to handle deeper feature maps)
    layers.Dropout(0.4),
    
    # Classification Head
    layers.Flatten(),
    
    # --- UPGRADED: Dense layer increased to 512 ---
    layers.Dense(512, activation='relu'),
    
    # --- NEW: Second Dropout layer ---
    layers.Dropout(0.3),
    
    layers.Dense(num_classes)
])

# ==========================================
# 4. COMPILE & TRAIN
# ==========================================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ENHANCEMENT 3: Smart Callbacks
# 1. Reduce Learning Rate if accuracy gets stuck
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,    
    patience=3,    
    min_lr=0.00001
)

# 2. Stop early if the model stops improving (Saves time)
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

print("\n--> Starting Training Process (Enhanced Deep CNN)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop]
)

# ==========================================
# 5. VISUALIZATION & SAVING
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()

# Save the model
model.save('new_stiffness_mapping_model_dataset2.h5')
print("\n--> Model saved as 'new_stiffness_mapping_model_dataset2.h5'")
print("--> Training Complete.")