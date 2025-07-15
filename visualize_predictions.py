import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np


# --- Configuration ---
BEST_MODEL_PATH = 'resnet18_transfer_best_model.pth'
ORIGINAL_DATA_PATH = 'dataset'
NUM_CLASSES = 5
IMAGE_SIZE = 224 #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_IMAGES_TO_SHOW = 9 

# --- Normalization Stats ---
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


print(f"Using device: {DEVICE}")
print(f"Loading model from: {BEST_MODEL_PATH}")
print(f"Loading images from: {ORIGINAL_DATA_PATH}")

# --- 1. Load Model Architecture and Weights ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {BEST_MODEL_PATH}"); exit()
except Exception as e:
    print(f"Error loading model weights: {e}"); exit()
model = model.to(DEVICE)
model.eval()

# --- 2. Define Validation Transforms---

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])
print("Using RGB validation transforms.")

try:
    temp_dataset = datasets.ImageFolder(ORIGINAL_DATA_PATH)
    class_names = temp_dataset.classes
    print(f"Class names: {class_names}")
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = [str(i) for i in range(NUM_CLASSES)]
    print(f"Using generic class names: {class_names}")

# --- 4. Function to Predict a Single Image ---
def predict_image(image_path, model, transform, device, class_names):
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None, None, None

    display_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    img_for_display = display_transform(img_pil)

    img_t = transform(img_pil) # transform incluye Resize, ToTensor, Normalize
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad(): out = model(batch_t)
    probabilities = F.softmax(out, dim=1)[0]
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    predicted_prob = probabilities[predicted_idx].item()

    return img_for_display, predicted_class, predicted_prob, probabilities.cpu().numpy()

# --- 5. Select and Predict Images ---
all_image_paths = []
for class_name in os.listdir(ORIGINAL_DATA_PATH):
    class_dir = os.path.join(ORIGINAL_DATA_PATH, class_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_image_paths.append(os.path.join(class_dir, img_file))

if not all_image_paths:
    print("Error: No images found in the original dataset directory.")
else:
    random_image_paths = random.sample(all_image_paths, min(NUM_IMAGES_TO_SHOW, len(all_image_paths)))

    # --- 6. Improved Plotting (3x3 Grid) ---
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12)) # Ajustar figsize para 3x3
    axes = axes.ravel() # Aplanar para iterar

    plot_index = 0
    for img_path in random_image_paths:
        if plot_index >= len(axes): break

        img_display, pred_class, pred_prob, all_probs = predict_image(img_path, model, val_transforms, DEVICE, class_names)

        if img_display is not None:
            true_class = os.path.basename(os.path.dirname(img_path))
            ax = axes[plot_index]
            ax.imshow(img_display)
            title_color = 'green' if pred_class == true_class else 'red'
            ax.set_title(f"True: {true_class}\nPred: {pred_class} ({pred_prob*100:.1f}%)", color=title_color, fontsize=9) # Reducir fuente si es necesario
            ax.axis('off')
            plot_index += 1

    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.94) # Ajustar espacio para el supertítulo
    plt.show()

    print("\nUse the toolbar in the Matplotlib window (usually magnifying glass) to zoom in on images.")