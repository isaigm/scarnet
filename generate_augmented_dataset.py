from torchvision import transforms, datasets
from PIL import Image
import os
from tqdm import tqdm


ORIGINAL_DATA_DIR = 'dataset'  
AUGMENTED_DATA_DIR = 'dataset_augmented_5x_RGB' #
IMAGES_PER_CLASS_TARGET = 250 
IMAGE_SIZE = 224

print(f"Generating RGB augmented dataset.")
print(f"Original data source: {ORIGINAL_DATA_DIR}")
print(f"Output directory: {AUGMENTED_DATA_DIR}")
print(f"Target images per class: {IMAGES_PER_CLASS_TARGET}")


augmentation_pipeline_rgb = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2), translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

final_save_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

if not os.path.exists(AUGMENTED_DATA_DIR):
    print(f"Creating directory: {AUGMENTED_DATA_DIR}")
    os.makedirs(AUGMENTED_DATA_DIR)
else:
    print(f"Output directory already exists: {AUGMENTED_DATA_DIR}")

try:
    original_dataset = datasets.ImageFolder(ORIGINAL_DATA_DIR)
    class_names = original_dataset.classes
    print(f"Found original dataset with {len(original_dataset)} images in {len(class_names)} classes.")
    print(f"Classes: {class_names}")
except FileNotFoundError:
    print(f"Error: Original dataset directory not found at {ORIGINAL_DATA_DIR}")
    exit()
except Exception as e:
    print(f"Error loading original dataset: {e}")
    exit()


print(f"\nStarting offline RGB augmentation to generate {IMAGES_PER_CLASS_TARGET} images per class...")

total_generated_count = 0
for class_idx, class_name in enumerate(class_names):
    print(f"\nProcessing class: {class_name}")
    original_class_dir = os.path.join(ORIGINAL_DATA_DIR, class_name)
    augmented_class_dir = os.path.join(AUGMENTED_DATA_DIR, class_name)

    if not os.path.exists(augmented_class_dir):
        os.makedirs(augmented_class_dir)

    original_image_files = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))] # Filtrar tipos comunes
    num_original_images = len(original_image_files)

    if num_original_images == 0:
        print(f"Warning: No images found in {original_class_dir}")
        continue

    num_augmentations_needed = IMAGES_PER_CLASS_TARGET - num_original_images
    if num_augmentations_needed < 0:
        print(f"Warning: Target ({IMAGES_PER_CLASS_TARGET}) is less than original count ({num_original_images}). Only copying originals.")
        num_augmentations_needed = 0

    augs_per_img = (num_augmentations_needed // num_original_images) + 1 if num_original_images > 0 else 0


    generated_count_for_class = 0

    pbar = tqdm(original_image_files, desc=f"Augmenting {class_name}")
    for filename in pbar:
        try:
            img_path = os.path.join(original_class_dir, filename)
            original_img = Image.open(img_path).convert('RGB')

            img_to_save = final_save_transform(original_img) # Aplicar resize final

            base_filename, _ = os.path.splitext(filename)
            save_filename_orig = f"{base_filename}_orig.png" # Usar PNG para consistencia
            img_to_save.save(os.path.join(augmented_class_dir, save_filename_orig))
            generated_count_for_class += 1
            total_generated_count += 1

            num_augs_generated_for_this_img = 0
            while num_augs_generated_for_this_img < augs_per_img and generated_count_for_class < IMAGES_PER_CLASS_TARGET:
                augmented_rgb = augmentation_pipeline_rgb(original_img)

                img_to_save = final_save_transform(augmented_rgb)

                save_filename_aug = f"{base_filename}_aug_{num_augs_generated_for_this_img}.png"
                img_to_save.save(os.path.join(augmented_class_dir, save_filename_aug))

                generated_count_for_class += 1
                total_generated_count += 1
                num_augs_generated_for_this_img += 1

            if generated_count_for_class >= IMAGES_PER_CLASS_TARGET:
                 pbar.n = len(original_image_files)
                 pbar.refresh()
                 break

        except Exception as e:
            print(f"\nError processing image {filename}: {e}")
            continue 

    print(f"Finished class {class_name}. Generated {generated_count_for_class} images.")


print(f"\nOffline RGB augmentation complete.")
print(f"Total images generated across all classes: {total_generated_count}")
print(f"Augmented data saved in: {AUGMENTED_DATA_DIR}")

print("\n--- Next Steps for Training Script ---")
print(f"1. Set DATASET_PATH_AUG = '{AUGMENTED_DATA_DIR}'")
print("2. For the 'train' DataLoader transforms, use ONLY:")
print("   - transforms.ToTensor()")
print("   - transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Or RGB stats calculated on your data")
print("3. For the 'val' DataLoader transforms, use:")
print("   - transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))")
print("   - transforms.ToTensor()")
print("   - transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Use same RGB stats")