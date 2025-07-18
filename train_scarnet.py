import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ==============================================================================
# --- Configuración Final (Modelo Ganador de 4 Clases) ---
# ==============================================================================
ORIGINAL_DATA_PATH = 'dataset' 
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4 # El modelo final y más robusto tiene 4 clases
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_VAL_SPLIT_RATIO = 0.8
TEST_SPLIT_RATIO = 0.2
VAL_FROM_TRAIN_RATIO = 0.15
BEST_MODEL_PATH = 'resnet_4class_randaugment_final_model.pth' # Nombre del modelo campeón

# --- Parámetros de Entrenamiento para Fine-Tuning en 2 Etapas ---
HEAD_LR = 1e-3
HEAD_EPOCHS = 25
FULL_LR = 1e-5
FULL_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1

print(f"Usando dispositivo: {DEVICE}")
print("Implementando la Estrategia Ganadora: Modelo de 4 Clases con RandAugment y Fine-Tuning Progresivo.")

# ==============================================================================
# --- Carga de Datos con Fusión de Clases y RandAugment ---
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Usamos RandAugment para la mejor estrategia de aumentos
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9), 
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.25)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Lógica de fusión de clases
temp_dataset = datasets.ImageFolder(ORIGINAL_DATA_PATH)
class_to_idx = temp_dataset.class_to_idx
boxcar_idx = class_to_idx.get('Boxcar')
rolling_idx = class_to_idx.get('Rolling')

if boxcar_idx is None or rolling_idx is None:
    print("Error: No se encontraron las clases 'Boxcar' o 'Rolling' en el dataset.")
    exit()

final_class_names = sorted([name for name in temp_dataset.classes if name != 'Rolling'])
final_class_to_idx = {name: i for i, name in enumerate(final_class_names)}
class_mapping = {}
for original_name, original_idx in class_to_idx.items():
    if original_name == 'Rolling': 
        new_idx = final_class_to_idx['Boxcar']
    else: 
        new_idx = final_class_to_idx[original_name]
    class_mapping[original_idx] = new_idx

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_folder = datasets.ImageFolder(root)
        self.transform = transform
        self.samples = self.image_folder.samples
        self.targets = [class_mapping[target] for _, target in self.samples]
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.image_folder.loader(path)
        target = self.targets[index]
        if self.transform: 
            sample = self.transform(sample)
        return sample, target
    def __len__(self):
        return len(self.samples)

full_dataset = CustomDataset(ORIGINAL_DATA_PATH)
total_size = len(full_dataset)
train_val_size = int(TRAIN_VAL_SPLIT_RATIO * total_size)
test_size = total_size - train_val_size
generator = torch.Generator().manual_seed(42)
train_val_subset_indices, test_subset_indices = random_split(range(total_size), [train_val_size, test_size], generator=generator)
train_size = int((1.0 - VAL_FROM_TRAIN_RATIO) * len(train_val_subset_indices))
val_size = len(train_val_subset_indices) - train_size
train_subset_indices, val_subset_indices = random_split(train_val_subset_indices, [train_size, val_size], generator=generator)

train_dataset = Subset(CustomDataset(ORIGINAL_DATA_PATH, transform=train_transforms), train_subset_indices.indices)
val_dataset = Subset(CustomDataset(ORIGINAL_DATA_PATH, transform=val_test_transforms), val_subset_indices.indices)
test_dataset = Subset(CustomDataset(ORIGINAL_DATA_PATH, transform=val_test_transforms), test_subset_indices.indices)

print(f"División de datos: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")
print(f"Resolviendo para 4 clases: {final_class_names}")

train_labels = [full_dataset.targets[i] for i in train_subset_indices.indices]
class_counts = np.bincount(train_labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = np.array([class_weights[label] for label in train_labels])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================================================================
# --- Modelo, Optimizador y Bucle de Entrenamiento ---
# ==============================================================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

# --- ETAPA 1: Entrenar el Cabezal Clasificador ---
print("\n--- Iniciando Etapa 1: Entrenando el Cabezal Clasificador ---")
for param in model.parameters(): 
    param.requires_grad = False
for param in model.fc.parameters(): 
    param.requires_grad = True
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)
for epoch in range(HEAD_EPOCHS):
    model.train()
    train_pbar = tqdm(train_loader, desc=f"Etapa 1, Epoch {epoch+1}/{HEAD_EPOCHS}")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# --- ETAPA 2: Fine-Tuning del Modelo Completo ---
print("\n--- Iniciando Etapa 2: Fine-Tuning del Modelo Completo ---")
for param in model.parameters(): 
    param.requires_grad = True
optimizer = optim.AdamW(model.parameters(), lr=FULL_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)
best_val_loss = float('inf')
epochs_no_improve = 0
for epoch in range(FULL_EPOCHS):
    model.train()
    running_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Etapa 2, Epoch {epoch+1}/{FULL_EPOCHS}")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_train_loss = running_loss / len(train_dataset)
    model.eval()
    running_val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
    
    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_acc = 100 * val_correct / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{FULL_EPOCHS} => Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        print(f"  Guardando mejor modelo en {BEST_MODEL_PATH}")
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping activado.")
        break
    print("-" * 50)

# ==============================================================================
# --- Evaluación Final ---
# ==============================================================================
print("\n--- Iniciando Evaluación Final en el Conjunto de Prueba ---")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluando en Test Set"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nPrecisión Final en Test: {accuracy_score(all_preds, all_labels) * 100:.2f}%")
print("\nReporte de Clasificación Final:")
print(classification_report(all_preds, all_labels, target_names=final_class_names, digits=4, zero_division=0))