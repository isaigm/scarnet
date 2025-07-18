import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math

# ==============================================================================
# --- Configuración ---
# ==============================================================================
MODEL_PATH = 'resnet_4class_randaugment_final_model.pth'
ORIGINAL_DATA_PATH = 'dataset'
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

print(f"Evaluando el modelo: {MODEL_PATH}")
print(f"Dataset: {ORIGINAL_DATA_PATH}")
print(f"Usando dispositivo: {DEVICE}")

# ==============================================================================
# --- Carga de Datos con Fusión de Clases ---
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

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

full_dataset = CustomDataset(ORIGINAL_DATA_PATH, transform=eval_transforms)
# DataLoader para la evaluación principal (sin shuffle para consistencia)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================================================================
# --- Cargar Modelo Entrenado ---
# ==============================================================================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==============================================================================
# --- Realizar Inferencia en Todo el Dataset ---
# ==============================================================================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(full_loader, desc="Procesando todo el dataset"):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==============================================================================
# --- Reporte de Rendimiento ---
# ==============================================================================
print(f"\n--- Reporte de Rendimiento en el Dataset Completo ---")
print(f"\nPrecisión General: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(all_labels, all_preds, target_names=final_class_names, digits=4, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=final_class_names, yticklabels=final_class_names)
plt.xlabel('Etiqueta Predicha', fontsize=12)
plt.ylabel('Etiqueta Real', fontsize=12)
plt.title('Matriz de Confusión en el Dataset Completo', fontsize=15)
plt.show()

# ==============================================================================
# --- Visualizar Muestra de Predicciones (ALEATORIO) ---
# ==============================================================================
def imshow(inp):
    """Función para des-normalizar y mostrar una imagen en un subplot de matplotlib."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

# --- ¡AQUÍ ESTÁ EL CAMBIO! ---
# Crear un nuevo DataLoader para la visualización con shuffle=True
# para obtener un lote aleatorio cada vez que se ejecuta.
vis_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Obtener un lote aleatorio de imágenes para visualizar
dataiter = iter(vis_loader)
images, labels = next(dataiter)

# Realizar predicciones en este lote
model.eval()
with torch.no_grad():
    outputs = model(images.to(DEVICE))
    _, preds = torch.max(outputs, 1)

# Crear una grilla para mostrar las imágenes
num_images_to_show = 9
rows = int(math.sqrt(num_images_to_show))
cols = int(math.sqrt(num_images_to_show))

plt.figure(figsize=(10, 10)) 

for i in range(num_images_to_show):
    if i >= len(images): break # Seguridad por si el lote es más pequeño
    ax = plt.subplot(rows, cols, i + 1)
    ax.axis('off')
    
    pred_class = final_class_names[preds[i]]
    true_class = final_class_names[labels[i]]
    
    is_correct = (pred_class == true_class)
    title_color = 'green' if is_correct else 'red'
    
    title = f"Pred: {pred_class}\nReal: {true_class}"
        
    ax.set_title(title, color=title_color, fontsize=10)
    imshow(images[i])

plt.tight_layout()
plt.show()