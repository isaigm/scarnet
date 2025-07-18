# Acne Scar Classification with Deep Learning

## A Replication and Optimization Study of the "ScarNet" Paper

This project presents a Deep Learning model capable of classifying 4 types of acne scars with a **final accuracy of 84.4%** on the full dataset.

The work began as an attempt to replicate the results of the paper "ScarNet: Development and Validation of a Novel Deep CNN Model" (IEEE Access). However, during the process, methodological challenges related to reproducibility and potential **data leakage** were identified.

As a result, the project evolved into a comparative analysis, culminating in the development of a more robust and reliable model based on **Transfer Learning**. This approach was demonstrated to be superior to training a CNN from scratch for this specific dataset.

## Final Model Features

- **Overall Accuracy:** 84.4% (evaluated on the full 250-image dataset).
- **Model:** ResNet18 with Transfer Learning.
- **Data Strategy:** A 4-class problem, merging the 'Rolling' and 'Boxcar' classes due to their high visual similarity.
- **Key Techniques:**
  - 2-Stage Progressive Fine-Tuning.
  - Advanced data augmentation with `RandAugment`.
  - Class imbalance handling with `WeightedRandomSampler`.

## Project Structure

```
/
|-- dataset/
|   |-- Boxcar/
|   |-- Hypertrophic/
|   |-- Ice Pick/
|   |-- Keloid/
|   `-- Rolling/
|-- resnet_4class_randaugment_final_model.pth  <-- The final trained model
|-- train_scarnet.py                           <-- Script to train the model from scratch
|-- test_model.py                              <-- Script to evaluate the model on the full dataset
|-- predict_single_image.py                    <-- Script to predict a single image
|-- requirements.txt                           <-- Project dependencies
`-- README.md
```

## Installation

It is recommended to use a virtual environment to run this project.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/isaigm/scarnet
    cd scarnet
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the Model
To train the model from scratch and replicate the results, simply run:
```bash
python train_scarnet.py
```
This script will train the model using the final strategy (ResNet18, 4 classes, etc.) and save the best-performing model as `resnet_4class_randaugment_final_model.pth`.

### 2. Evaluate the Trained Model
To evaluate the provided `resnet_4class_randaugment_final_model.pth` on the entire dataset, run:
```bash
python test_model.py
```
This will print the classification report and overall accuracy to the console, and it will display a confusion matrix and a sample of predictions.

### 3. Predict a Single Image
To use the model to classify a new image, use the `predict_single_image.py` script and pass the path to the image:
```bash
python predict_single_image.py --image "path/to/your/image.jpg"
```
For example:
```bash
python predict_single_image.py --image "dataset/Ice Pick/icepick (1).png"
```

## Final Results

The final model, trained with the 4-class strategy, achieved an **overall accuracy of 84.4%** on the full dataset.

#### Classification Report:
| Class        | Precision | Recall | F1-Score | Support |
| :----------- | :-------- | :----- | :------- | :------ |
| **Boxcar**   | 0.9718    | 0.6970 | 0.8118   | 99      |
| **Hypertrophic** | 0.9000    | 0.9000 | 0.9000   | 50      |
| **Ice Pick** | 0.6712    | 0.9800 | 0.7967   | 50      |
| **Keloid**   | 0.8571    | 0.9412 | 0.8972   | 51      |
|--------------|-----------|--------|----------|---------|
| **Macro Avg**| 0.8501    | 0.8795 | 0.8514   | 250     |
| **Weighted Avg**| 0.8739 | 0.8440 | 0.8438 | 250 |

## Methodology and Key Findings

-   **Irreproducibility of the Original Paper:** It was concluded that the results from the `ScarNet` paper are likely a product of **data leakage** in their pre-augmentation methodology. This is a common pitfall that leads to inflated and non-generalizable results.
-   **Ineffectiveness of Training from Scratch:** Multiple attempts to train a CNN from scratch (even with robust architectures and advanced techniques) proved ineffective due to the **extreme scarcity of data**, resulting in severe overfitting.
-   **Superiority of Transfer Learning:** A **Transfer Learning** approach on the standard **RGB** color space was scientifically demonstrated to be a fundamentally superior strategy for this problem.
-   **Class Merging as a Strategic Decision:** The 'Rolling' scar class could not be learned independently due to its high visual similarity to 'Boxcar' and the lack of clear, discriminative features in the dataset. Merging these two classes was a key decision to achieve a robust and well-balanced model.

## License

This project is distributed under the MIT License.