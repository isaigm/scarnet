import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Configuration ---
BEST_MODEL_PATH = 'resnet18_transfer_best_model.pth'
ORIGINAL_DATA_PATH = 'dataset'
NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")
print(f"Loading model from: {BEST_MODEL_PATH}")
print(f"Evaluating on full dataset from: {ORIGINAL_DATA_PATH}")

# --- 1. Load Model Architecture and Weights ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model weights loaded successfully.")

# --- 2. Define Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])
print("Using RGB transforms for evaluation.")

# --- 3. Prepare Full Dataset as Test ---
full_dataset = datasets.ImageFolder(ORIGINAL_DATA_PATH, transform=transform)
total_images = len(full_dataset)
class_names = full_dataset.classes
print(f"Total images loaded: {total_images}, Classes: {class_names}")

# --- 4. DataLoader for Full Dataset ---
test_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# --- 5. Get Predictions ---
all_preds = []
all_labels = []
print("\nGetting predictions on the full dataset...")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- 6. Metrics and Visualization ---
print("\n--- Evaluation Metrics ---")
# Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
# Cohen's Kappa
kappa = cohen_kappa_score(all_labels, all_preds)
print(f"Cohen's Kappa: {kappa:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Specificity per class
def calc_specificity(cm, idx):
    tn = cm.sum() - (cm[idx, :].sum() + cm[:, idx].sum() - cm[idx, idx])
    fp = cm[:, idx].sum() - cm[idx, idx]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

print("\nSpecificity per class:")
for i, name in enumerate(class_names):
    spec = calc_specificity(cm, i)
    print(f"  - {name}: {spec:.4f}")

print("\n--- End of Evaluation ---")
