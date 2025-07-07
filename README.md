# Chest X-Ray Image Classification using Deep Learning

This project focuses on building and evaluating deep learning models to classify chest X-ray images into three categories:

* **Normal**
* **Lung Opacity**
* **Viral Pneumonia**

It leverages Convolutional Neural Networks (CNNs) and Transfer Learning with pre-trained architectures like **ResNet50** and **MobileNet** to detect potential signs of pneumonia or other lung abnormalities in chest radiographs.

##  Dataset

* **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* Contains grayscale chest X-ray images categorized into the three classes.
* The dataset was split using an 80/20 training-validation split.

##  Preprocessing

* Resized all images to **224x224**
* Pixel rescaling (`1./255`)
* Data augmentation (rotation, zoom, horizontal flip)
* One-hot encoding for categorical labels

## Models Implemented

### 1. **Custom CNN**

* Basic Conv2D-ReLU-MaxPooling architecture
* Achieved \~82% validation accuracy

### 2. **ResNet50**

* Transfer learning with frozen base layers
* Fine-tuned top layers
* Validation Accuracy: **\~89%**
* ROC AUC Score: **0.97**

### 3. **MobileNet** *(Best Performing)*

* Lightweight and fast training
* Validation Accuracy: **\~91%**
* ROC AUC Score: **0.978**
* Best balance between speed and accuracy

##  Model Performance Summary

| Model         | Val Accuracy | ROC AUC   | Precision | Recall   | F1 Score |
| ------------- | ------------ | --------- | --------- | -------- | -------- |
| Custom CNN    | 0.82         | -         | 0.82      | 0.82     | 0.81     |
| ResNet50      | 0.89         | 0.97      | 0.89      | 0.89     | 0.88     |
| **MobileNet** | **0.91**     | **0.978** | **0.91**  | **0.91** | **0.90** |

## Key Insights

* MobileNet achieved the best overall performance with the highest ROC AUC and accuracy.
* Custom CNN performed well but was outperformed by pre-trained models.
* ROC AUC was crucial in evaluating multiclass classification beyond accuracy.

##  Limitations

* Computational constraints limited training time and GPU usage.
* No access to a large clinical dataset for testing generalization.
* Fine-tuning was limited to top layers due to hardware restrictions.

##  Next Steps

* Deploy model using TensorFlow Lite or Flask for real-world usage.
* Collect more diverse data and explore model explainability (e.g., Grad-CAM).
* Experiment with ensemble models and more tuning of hyperparameters.

## Code

All code is available in this repository, including:

* Model training scripts
* Evaluation metrics
* Visualization (accuracy/loss plots, classification reports)
* Optional: Pretrained weights (if applicable)

---


