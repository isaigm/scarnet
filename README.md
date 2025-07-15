# ScarNet: Acne Scar Classification

This repository contains the code for classifying acne scars using a Deep Learning model. The project is based on the dataset and concept from the paper "ScarNet: Development and Validation of a Novel Deep CNN Model for Acne Scar Classification With a New Dataset", but with an adapted implementation due to an error in the original architecture that made exact replication impossible.

## Project Overview

The goal of this project is to classify different types of acne scars from images. Due to an issue in the original paper's architecture, we opted for a **Transfer Learning** approach using a pre-trained model (ResNet18) on ImageNet, and fine-tuned it for this specific task.

## Dataset

The dataset is organized under the `dataset/` directory and contains the following acne scar categories:

* `Boxcar/`
* `Hypertrophic/`
* `Ice Pick/`
* `Keloid/`
* `Rolling/`

This dataset was adapted from the paper "ScarNet: Development and Validation of a Novel Deep CNN Model for Acne Scar Classification With a New Dataset".

## Model Architecture and Training

Because of a discrepancy in the original paper's model description, we implemented a **Transfer Learning** approach. We used **ResNet18** pre-trained on ImageNet and replaced its classifier head to adapt it for acne scar classification.

Training is performed using the `train_scarnet.py` script. Our transfer learning experiments yielded an accuracy between **83% and 89%**. The confusion matrix from testing shows that the model performs robustly without significant bias toward any class.

## Repository Structure

* `generate_augmented_dataset.py`: Script for generating an augmented dataset (if applicable).
* `resnet18_transfer_best_model.pth`: The fine-tuned ResNet18 model saved in PyTorch format.
* `test_model.py`: Script for evaluating the model's performance.
* `train_scarnet.py`: Main script for training the model.
* `visualize_predictions.py`: Script to visualize model predictions.
* `dataset/`: Directory containing the images organized by scar type.
* `requirements.txt`: Python dependencies for the project.

## Getting Started

Follow these steps to set up and run the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/isaigm/scarnet
   cd scarnet
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/macOS
   # venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**

   ```bash
   python train_scarnet.py
   ```

   This will save the best model as `resnet18_transfer_best_model.pth`.

5. **Evaluate the model:**

   ```bash
   python test_model.py
   ```

6. **Visualize predictions:**

   ```bash
   python visualize_predictions.py
   ```

## Performance

The transfer learning approach (ResNet18) achieved the following results on the full dataset of 250 images:

* **Overall Accuracy:** 83.20%
* **Cohen’s Kappa:** 0.79

### Detailed Metrics

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Boxcar**       | 0.7800    | 0.7800 | 0.7800   | 50      |
| **Hypertrophic** | 0.8627    | 0.8800 | 0.8713   | 50      |
| **Ice Pick**     | 0.7917    | 0.7600 | 0.7755   | 50      |
| **Keloid**       | 0.8704    | 0.9216 | 0.8952   | 51      |
| **Rolling**      | 0.8511    | 0.8163 | 0.8333   | 49      |
| **Macro Avg**    | 0.8312    | 0.8316 | 0.8311   | 250     |
| **Weighted Avg** | 0.8312    | 0.8320 | 0.8313   | 250     |

The confusion matrix is as follows:

```
[[39  1  7  1  2]
 [ 3 44  0  3  0]
 [ 6  0 38  1  5]
 [ 0  4  0 47  0]
 [ 2  2  3  2 40]]
```

Specificity per class ranged from 0.945 to 0.965, indicating balanced performance across categories.

These results reflect that while overall performance remains strong, some confusion exists between particular classes (e.g., Boxcar vs. Ice Pick).

**Note:** In a previous experiment—where the 5× augmented dataset was freshly regenerated—the model achieved even higher accuracy (up to 87%), underscoring the impact of thorough data augmentation on performance. Consider targeted data augmentation or model calibration to improve recall and precision on these pairs.


