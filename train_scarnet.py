import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets, models 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


DATASET_PATH_AUG = 'dataset_augmented_5x_RGB' #
ORIGINAL_DATA_PATH = 'dataset' #

# --- Training Hyperparameters ---
IMAGE_SIZE = 224
BATCH_SIZE = 32  
LEARNING_RATE = 0.001 
NUM_EPOCHS = 50   
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Regularization and Optimization ---
WEIGHT_DECAY = 1e-4 
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1
EARLY_STOPPING_PATIENCE = 10
BEST_MODEL_PATH = 'resnet18_transfer_best_model.pth'

print(f"Using device: {DEVICE}")
print(f"Performing Transfer Learning with ResNet18.")
print(f"Training on pre-augmented RGB dataset: {DATASET_PATH_AUG}")
print(f"Validation data from: {ORIGINAL_DATA_PATH}")

# --- Use Standard RGB Normalization ---
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

print(f"Using ImageNet RGB Normalization Mean: {RGB_MEAN}")
print(f"Using ImageNet RGB Normalization Std: {RGB_STD}")

# --- Define Transforms (Simple RGB, como en el intento anterior) ---
# --- Training Transforms (SIMPLIFIED for pre-augmented RGB) ---
train_transforms_preaug_rgb = transforms.Compose([
    # Asume que el pre-aumentador ya hizo Resize a IMAGE_SIZE
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])

# --- Validation Transforms (Simple RGB for original validation data) ---
val_transforms_rgb = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Redimensionar originales de validación
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD) # Usar mismas stats RGB
])

# --- Load Data (Igual que en el intento anterior con RGB pre-aumentado) ---
# --- Training Data ---
try:
    train_dataset = datasets.ImageFolder(DATASET_PATH_AUG, transform=train_transforms_preaug_rgb)
    print(f"Loaded training dataset from: {DATASET_PATH_AUG} with {len(train_dataset)} images.")
    if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
except FileNotFoundError: print(f"Error: Training dataset not found at {DATASET_PATH_AUG}"); exit()
except Exception as e: print(f"Error loading training dataset: {e}"); exit()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# --- Validation Data (Split from original) ---
VAL_SPLIT_RATIO = 0.2
try:
    original_full_dataset = datasets.ImageFolder(ORIGINAL_DATA_PATH)
    original_total_size = len(original_full_dataset)
    if original_total_size == 0: raise ValueError("Original dataset empty.")
    val_size = int(VAL_SPLIT_RATIO * original_total_size); val_size = max(val_size, NUM_CLASSES)
    if val_size >= original_total_size:
        print("Warning: Using all original data for validation."); val_size = original_total_size
        _ , val_indices_orig_indices = [], list(range(original_total_size))
        val_indices_orig = type('obj', (object,), {'indices': val_indices_orig_indices})()
    else:
        _ , val_indices_orig = random_split(range(original_total_size), [original_total_size - val_size, val_size])
    val_subset_orig = Subset(original_full_dataset, val_indices_orig.indices)
    class DatasetFromSubset(torch.utils.data.Dataset): # Wrapper
        def __init__(self, subset, transform=None): self.subset = subset; self.transform = transform
        def __getitem__(self, index): x, y = self.subset[index]; return self.transform(x) if self.transform else x, y
        def __len__(self): return len(self.subset)
    val_dataset = DatasetFromSubset(val_subset_orig, transform=val_transforms_rgb)
    print(f"Created validation dataset from: {ORIGINAL_DATA_PATH} with {len(val_dataset)} images.")
    if len(val_dataset) == 0: print("Warning: Validation dataset is empty.")
except FileNotFoundError: print(f"Error: Original dataset not found at {ORIGINAL_DATA_PATH}"); exit()
except Exception as e: print(f"Error creating validation dataset: {e}"); exit()
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True) if len(val_dataset) > 0 else None
if val_loader is None: print("Warning: Validation disabled.")

# --- Model Setup: Transfer Learning with ResNet18 ---

model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES) 


model_ft = model_ft.to(DEVICE)

print("ResNet18 loaded. Feature extractor frozen. Final layer replaced.")

params_to_optimize = model_ft.fc.parameters()

# --- Loss, Optimizer, Scheduler ---
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True) if val_loader else None

# --- Training Loop ---
print("\nStarting Transfer Learning Training...")
model = model_ft 

train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_triggered = False
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train() # 
    running_loss = 0.0
    train_correct = 0
    train_total = len(train_dataset)
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for i, (inputs, labels) in enumerate(train_pbar):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.1E}'})
    epoch_train_loss = running_loss / train_total if train_total > 0 else 0
    epoch_train_acc = 100 * train_correct / train_total if train_total > 0 else 0
    train_losses.append(epoch_train_loss)

    # --- Validation Phase ---
    epoch_val_loss = float('nan'); epoch_val_acc = float('nan')
    if val_loader:
        model.eval()
        running_val_loss = 0.0; val_correct = 0; val_total = len(val_dataset)
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_val_loss = running_val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(epoch_val_loss); val_accuracies.append(epoch_val_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} => Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # --- Scheduler, Checkpointing, Early Stopping ---
        if scheduler: scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss and val_total > 0:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0
        elif val_total > 0:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")
        if val_total > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs."); early_stop_triggered = True; break
    else:
         print(f"Epoch {epoch+1}/{NUM_EPOCHS} => Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | (No validation set)")
         if epoch == NUM_EPOCHS - 1:
             print("Saving final model state."); torch.save(model.state_dict(), BEST_MODEL_PATH.replace('_best_', '_final_'))
    print("-" * 50)

# --- End of Training ---

end_time = time.time(); total_time = end_time - start_time
print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
if not early_stop_triggered and val_loader: print('Finished Training (reached max epochs).')
elif early_stop_triggered: print('Finished Training (early stopped).')
else: print('Finished Training (no validation performed).')

if val_loader and os.path.exists(BEST_MODEL_PATH):
    print(f"Loading best model weights from {BEST_MODEL_PATH} with val_loss: {best_val_loss:.4f}")
    try: model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    except Exception as e: print(f"Error loading best model weights: {e}")
elif not val_loader and os.path.exists(BEST_MODEL_PATH.replace('_best_', '_final_')):
     print(f"Loading final model weights from {BEST_MODEL_PATH.replace('_best_', '_final_')}")
     try: model.load_state_dict(torch.load(BEST_MODEL_PATH.replace('_best_', '_final_'))); model.eval()
     except Exception as e: print(f"Error loading final model weights: {e}")
else: print("No best/final model was saved. Using last model state."); model.eval()


# --- Plotting ---
if val_losses:
    best_epoch_idx = len(val_losses) - (epochs_no_improve + 1 if early_stop_triggered else 0)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if train_losses: plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    if best_epoch_idx >= 0: plt.axvline(best_epoch_idx + 1, color='r', linestyle='--', label=f'Best Model Epoch ({best_epoch_idx+1})')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    if val_accuracies: plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    if best_epoch_idx >= 0: plt.axvline(best_epoch_idx + 1, color='r', linestyle='--', label=f'Best Model Epoch ({best_epoch_idx+1})')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.title('Validation Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()
elif train_losses:
    plt.figure(figsize=(7, 5)); plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend(); plt.grid(True); plt.show()
else: print("No results to plot.")

print("\n--- Note ---")
print("Transfer Learning training finished.")
print("Compare the validation accuracy achieved with previous results.")