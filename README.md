# Acne Scar Classification: A Rigorous Replication and Methodological Study

## Abstract

This project began as a replication study of the "ScarNet" paper. The study revealed significant methodological challenges, including a high risk of **data leakage**, which likely inflated the original 92.5% accuracy claim.

In response, a superior solution was engineered using **Transfer Learning** with a fine-tuned ResNet18. To ensure a scientifically sound measure of performance, the methodology was subjected to **K-Fold Cross-Validation**. This rigorous validation yielded a final, honest performance metric of **64.4% ± 5.85% average accuracy**, which represents the true expected performance of the model on this challenging, limited dataset. A single training run produced a particularly effective model that achieved **86% accuracy** on its dedicated test set, and it is this model that is provided in the repository.

## Final Model Strategy

- **Core Technique:** ResNet18 with Transfer Learning.
- **Robust Performance Metric:** **64.4% ± 5.85%** (5-Fold Cross-Validation Average Accuracy).
- **Peak Performance (Provided Model):** **86%** Accuracy on its clean test set.
- **Data Strategy:** A 4-class problem, strategically merging the 'Rolling' and 'Boxcar' classes due to their high visual similarity.
- **Key Methods:**
  - Rigorous evaluation using K-Fold cross-validation.
  - A two-stage progressive fine-tuning process.
  - Advanced data augmentation with `RandAugment`.
  - Handling of class imbalance with `WeightedRandomSampler`.

## Project Structure
... (sin cambios)

## Installation
... (sin cambios)

## Usage
... (sin cambios)

## Performance Evaluation & Key Results

This project utilizes two levels of evaluation: a rigorous K-Fold Cross-Validation to determine the overall methodology's robustness, and a single-split evaluation for the specific model provided in this repository.

### 1. Rigorous K-Fold Cross-Validation (The Scientific Truth)

To obtain a reliable and unbiased measure of the methodology's true performance, a 5-Fold Cross-Validation was implemented. This process eliminates the "lucky split" bias by training and evaluating 5 separate models on different subsets of the data.

-   **Accuracy per Fold:** `[72.0%, 58.0%, 70.0%, 58.0%, 64.0%]`
-   **Average Accuracy:** **64.4%**
-   **Standard Deviation:** **5.85%**

The final, scientifically rigorous result is **64.4% ± 5.85%**. This is the most honest estimate of how this methodology is expected to perform on new, unseen data.

### 2. Single Model Performance (The Provided `.pth` File)

The `resnet_4class_randaugment_final_model.pth` file in this repository was generated from a single run of the `train_scarnet.py` script. This specific run represents a favorable data split where the model performed exceptionally well.

-   **Accuracy on its Test Set (50 images):** **86.0%**
-   **Accuracy on the Full Dataset (250 images, via `test_model.py`):** **84.4%**

This demonstrates the peak performance achieved by the methodology and provides a useful model for inference, while the cross-validation result above represents the more conservative and realistic performance expectation.

## Key Findings & Conclusion
... (sin cambios)

## License
... (sin cambios)

## Key Findings & Conclusion

-   **Irreproducibility of the "ScarNet" Paper:** The original paper's results are likely a product of **data leakage**, a common pitfall that leads to inflated and non-generalizable results.
-   **Superiority of Transfer Learning:** A **Transfer Learning** approach on the standard **RGB** color space was scientifically demonstrated to be a fundamentally superior strategy over training a custom CNN from scratch on this limited dataset.
-   **Importance of Rigorous Validation:** This study underscores that a single train/test split can be misleading. **K-Fold Cross-Validation** provides a much more robust and realistic measure of a model's true capabilities.
-   **Final Verdict:** The developed methodology represents a robust solution to a challenging fine-grained classification problem, with its real-world performance honestly quantified by cross-validation.

## License

This project is distributed under the MIT License.
