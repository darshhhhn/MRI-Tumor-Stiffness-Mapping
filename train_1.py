import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Your dataset path
DATASET_PATH = r"C:\Users\Darshan\Documents\vscode\Machine Learning Project\Dataset 1"

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 50 

print("--> Initializing Project: Tissue Stiffness Mapping (via Tumor Classification)")
print(f"--> Loading Data from: {DATASET_PATH}")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,       
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\n--> Classes Found: {class_names}")

except Exception as e:
    print(f"\n[ERROR] Could not load dataset. Check your path.\nError: {e}")
    exit()

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization Layer
normalization_layer = layers.Rescaling(1./255)

# Data Augmentation Block
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
    
    # Convolutional Block 1 (Basic features like edges)
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 2 (Textures)
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 3 (Basic shapes)
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # --- NEW: Convolutional Block 4 (Complex structural patterns) ---
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Dropout (Increased to 0.4 because the network is deeper)
    layers.Dropout(0.4),
    
    # Flatten & Dense Layers
    layers.Flatten(),
    
    # --- UPGRADED: Dense layer increased to 512 to handle the new deeper features ---
    layers.Dense(512, activation='relu'),
    
    # --- NEW: Second Dropout layer to force the final decision neurons to generalize ---
    layers.Dropout(0.3),
    
    layers.Dense(num_classes)
])

# ==========================================
# 4. COMPILE & TRAIN
# ==========================================

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Callbacks for Smart Training
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,    
    patience=3,    
    min_lr=0.00001
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
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
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model
model.save('new_stiffness_mapping_model.h5')
print("\n--> Model saved as 'new_stiffness_mapping_model.h5'")
print("--> Training Complete.")