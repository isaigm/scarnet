import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold # ¡Importante!

# ==============================================================================
# --- Configuración ---
# ==============================================================================
ORIGINAL_DATA_PATH = 'dataset' 
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5 # Definimos el número de pliegues para la validación cruzada

# --- Parámetros de Entrenamiento ---
HEAD_LR = 1e-3
HEAD_EPOCHS = 25
FULL_LR = 1e-5
FULL_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1

print(f"Iniciando Validación Cruzada de {K_FOLDS} pliegues...")

# ==============================================================================
# --- Carga de Datos y Lógica de Fusión (sin cambios) ---
# ==============================================================================
# (El código para las transformaciones, la fusión de clases y el CustomDataset
# es idéntico al del script anterior. Se omite por brevedad, pero debe estar aquí)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandAugment(num_ops=2, magnitude=9), 
    transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.25)
])
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
temp_dataset = datasets.ImageFolder(ORIGINAL_DATA_PATH)
class_to_idx = temp_dataset.class_to_idx
boxcar_idx = class_to_idx.get('Boxcar'); rolling_idx = class_to_idx.get('Rolling')
final_class_names = sorted([name for name in temp_dataset.classes if name != 'Rolling'])
final_class_to_idx = {name: i for i, name in enumerate(final_class_names)}
class_mapping = {}
for original_name, original_idx in class_to_idx.items():
    if original_name == 'Rolling': new_idx = final_class_to_idx['Boxcar']
    else: new_idx = final_class_to_idx[original_name]
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
        if self.transform: sample = self.transform(sample)
        return sample, target
    def __len__(self):
        return len(self.samples)

full_dataset = CustomDataset(ORIGINAL_DATA_PATH)
dataset_targets = full_dataset.targets

# ==============================================================================
# --- Bucle Principal de Validación Cruzada ---
# ==============================================================================
kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
test_accuracies = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset, dataset_targets)):
    print(f"\n{'='*20} PLIEGUE {fold + 1}/{K_FOLDS} {'='*20}")
    
    # --- Crear Subsets y DataLoaders para este pliegue ---
    train_dataset = Subset(CustomDataset(ORIGINAL_DATA_PATH, transform=train_transforms), train_ids)
    test_dataset = Subset(CustomDataset(ORIGINAL_DATA_PATH, transform=val_test_transforms), test_ids)
    
    train_labels = [dataset_targets[i] for i in train_ids]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Re-inicializar el Modelo y Optimizador (¡MUY IMPORTANTE!) ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # --- ETAPA 1: Entrenar el Cabezal Clasificador ---
    print("\n--- Iniciando Etapa 1: Entrenando el Cabezal Clasificador ---")
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)
    for epoch in range(HEAD_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
    
    # --- ETAPA 2: Fine-Tuning del Modelo Completo (SIN VALIDACIÓN INTERNA) ---
    # Nota: En un K-Fold riguroso, el EarlyStopping se haría en un sub-split de validación.
    # Para simplificar, entrenaremos por un número fijo de épocas.
    print("\n--- Iniciando Etapa 2: Fine-Tuning del Modelo Completo ---")
    for param in model.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=FULL_LR)
    for epoch in range(FULL_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()

    # --- Evaluación del Pliegue ---
    print("\n--- Evaluando el pliegue... ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs); _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    
    fold_accuracy = accuracy_score(all_labels, all_preds) * 100
    test_accuracies.append(fold_accuracy)
    print(f"Precisión del Pliegue {fold + 1}: {fold_accuracy:.2f}%")

# ==============================================================================
# --- Resultados Finales de la Validación Cruzada ---
# ==============================================================================
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print(f"\n{'='*20} RESULTADO FINAL DE LA VALIDACIÓN CRUZADA {'='*20}")
print(f"Precisión en cada pliegue: {[f'{acc:.2f}%' for acc in test_accuracies]}")
print(f"Precisión Promedio: {mean_accuracy:.2f}%")
print(f"Desviación Estándar: {std_accuracy:.2f}%")
print(f"\nResultado Final: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

# ==============================================================================
# --- ¿Y ahora qué? ---
# ==============================================================================
print("\nValidación de la metodología completada.")
print("Para crear el modelo final de producción, entrena el script original en TODO el dataset.")
