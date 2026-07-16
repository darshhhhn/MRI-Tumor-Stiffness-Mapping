import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Dataset and Model Paths
MODEL_PATH = 'new_stiffness_mapping_model_dataset2.h5'
TEST_DIR = r"C:\Users\Darshan\Documents\vscode\Machine Learning Project\Dataset 2\TumourClassificationImages\Test"

# NEW: Custom Folder for Mode 2
CUSTOM_TEST_DIR = r"C:\Users\Darshan\Documents\vscode\Machine Learning Project\test"

# Explicit Class Names (Must match training alphabetical order)
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary']

# Stiffness Mapping Logic
STIFFNESS_MAP = {
    'glioma':      'HIGH Stiffness (Hard)',
    'meningioma':  'MEDIUM Stiffness (Firm)',
    'pituitary':   'LOW Stiffness (Soft)',
    'notumor':     'NORMAL Stiffness (Healthy)',
    'no_tumor':    'NORMAL Stiffness (Healthy)'
}

BATCH_SIZE = 32
IMG_SIZE = (180, 180)

# ==========================================
# 2. DATA LOADING LOGIC (MODE 1)
# ==========================================
def load_resources():
    print(f"\n--> Initializing Evaluation...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test Folder not found at {TEST_DIR}")
        
    print("--> Loading Trained Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"--> Loading Test Data from: {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False # False is required to keep labels aligned for reports
    )
    
    return model, test_ds

# ==========================================
# 3. METRICS EVALUATION (CM, Report, ROC) - MODE 1
# ==========================================
def run_evaluation(model, dataset):
    class_names = dataset.class_names
    print(f"--> Classes detected: {class_names}")
    print("--> Predicting all images (This may take a moment)...")

    y_true = []
    y_pred = []
    y_pred_probs = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_ids = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_ids)
        y_pred_probs.extend(predictions)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.close('all') 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('ML Model Predicted Diagnosis')
    plt.title('Confusion Matrix')
    plt.show()

    # --- 2. ROC Curve ---
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = len(class_names)
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # --- 3. Text Report ---
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return class_names

# ==========================================
# 4. GRAD-CAM (Tumor Highlighting Logic)
# ==========================================
def get_gradcam_heatmap(img_array, model):
    last_conv_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i

    if last_conv_idx is None:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1]))

    with tf.GradientTape() as tape:
        x = img_array
        for i in range(last_conv_idx + 1):
            x = model.layers[i](x, training=False) 
            
        conv_outputs = x
        tape.watch(conv_outputs)
        
        for i in range(last_conv_idx + 1, len(model.layers)):
            x = model.layers[i](x, training=False)
            
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) 
    return heatmap.numpy()

# ==========================================
# 5. VISUALIZATION ENGINE (Shared by both modes)
# ==========================================
def plot_tumor_highlight(img_tensor, actual_class, pred_class_name, confidence, stiffness_label, heatmap_resized):
    plt.close('all') 
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # --- LEFT: Original MRI ---
    axes[0].imshow(img_tensor[0].numpy().astype("uint8"))
    title_text = f"Original Scan"
    if actual_class:
        title_text += f"\nActual: {actual_class.upper()}"
    axes[0].set_title(title_text, fontsize=12)
    axes[0].axis("off")
    
    # --- RIGHT: Highlighted Tumor ---
    axes[1].imshow(img_tensor[0].numpy().astype("uint8"))
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4) 
    axes[1].contour(heatmap_resized, levels=[0.5, 0.8], colors=['white', 'red'], linewidths=2)
    
    axes[1].set_title(f"ML Prediction: {pred_class_name.upper()}\nConfidence: {confidence:.2f}%", fontsize=12)
    axes[1].axis("off")
    
    # Red Stiffness Banner 
    axes[1].text(5, 20, f"STIFFNESS MAP: {stiffness_label}", 
                 fontsize=10, color='white', weight='bold', backgroundcolor='red',
                 bbox=dict(facecolor='red', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. MODE LOGIC EXECUTION
# ==========================================

def run_mode_1_default():
    print("\n[MODE 1 INITIATED: Full Dataset Evaluation]")
    model, test_ds = load_resources()
    classes = run_evaluation(model, test_ds)
    
    print("\n--> Generating 'Stiffness Map' Demo Images for 3 random classes...")
    demo_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, seed=None, image_size=IMG_SIZE, batch_size=1, shuffle=True)
    
    found_classes = set()
    demo_images = []
    
    for image, label in demo_ds:
        actual_class = classes[int(label[0].numpy())]
        if actual_class not in found_classes:
            found_classes.add(actual_class)
            demo_images.append((image, actual_class))
        if len(found_classes) == len(classes):
            break

    for img_tensor, actual_class in demo_images:
        prediction = model.predict(img_tensor, verbose=0)
        score = tf.nn.softmax(prediction[0])
        pred_class_idx = np.argmax(score)
        pred_class_name = classes[pred_class_idx]
        confidence = 100 * np.max(score)
        stiffness_label = STIFFNESS_MAP.get(pred_class_name.lower(), "Unknown Stiffness")
        
        heatmap = get_gradcam_heatmap(img_tensor, model)
        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:, :, 0]
        
        plot_tumor_highlight(img_tensor, actual_class, pred_class_name, confidence, stiffness_label, heatmap_resized)
        print(f"--> Demo Displayed: {pred_class_name} ({stiffness_label})")


def run_mode_2_custom_folder():
    print(f"\n[MODE 2 INITIATED: Custom Folder Testing]")
    print(f"--> Scanning Folder: {CUSTOM_TEST_DIR}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return
        
    if not os.path.exists(CUSTOM_TEST_DIR):
        print(f"[ERROR] The custom folder does not exist: {CUSTOM_TEST_DIR}")
        print("Please create it and paste some images inside.")
        return

    # Find all images in the folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(CUSTOM_TEST_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"[WARNING] No image files found in {CUSTOM_TEST_DIR}.")
        print("Please paste some MRI images into this folder and try again.")
        return

    print(f"--> Found {len(image_files)} images. Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    for file_name in image_files:
        img_path = os.path.join(CUSTOM_TEST_DIR, file_name)
        print(f"\n--> Processing: {file_name}")
        
        # Load and preprocess the single image
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_tensor = tf.expand_dims(img_array, 0) # Add batch dimension
        
        # Predict
        prediction = model.predict(img_tensor, verbose=0)
        score = tf.nn.softmax(prediction[0])
        
        pred_class_idx = np.argmax(score)
        pred_class_name = CLASS_NAMES[pred_class_idx]
        confidence = 100 * np.max(score)
        stiffness_label = STIFFNESS_MAP.get(pred_class_name.lower(), "Unknown Stiffness")
        
        # Heatmap
        heatmap = get_gradcam_heatmap(img_tensor, model)
        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:, :, 0]
        
        # Display (Since we don't know the actual class for random pasted images, we pass None)
        plot_tumor_highlight(img_tensor, None, pred_class_name, confidence, stiffness_label, heatmap_resized)

# ==========================================
# 7. MAIN INTERACTIVE MENU
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print(" MRI TUMOR CLASSIFICATION & STIFFNESS MAPPING")
    print("="*50)
    print("Please select an evaluation mode:")
    print("  [1] Academic Mode (Runs Full Evaluation, Matrices, and 3 Demos)")
    print("  [2] Clinical Mode (Tests loose images pasted in your Custom Folder)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    try:
        if choice == '1':
            run_mode_1_default()
        elif choice == '2':
            run_mode_2_custom_folder()
        else:
            print("[ERROR] Invalid selection. Please restart and enter 1 or 2.")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")