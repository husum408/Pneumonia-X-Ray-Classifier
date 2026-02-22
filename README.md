# Pneumonia Detection for Chest X-Rays Using CNNs
This project uses a convolutional neural network for binary classification of chest X-ray scans as healthy or indicating pneumonia. Using a publicly available dataset, I trained and evaluated a classifier and analyzed the model's behavior using tools such as Grad-CAM. This project was completed as an educational exploration of medical image classification and interpretability.

# Motivation
Classifiers are useful in medical imaging as tools for diagnosing patients. A healthy patient's lungs will appear dark in a chest x-ray, while a patient with pneumonia's lungs will show white patches due to fluid buildup. This difference can be readily analyzed using computer vision models.

Grad-CAM was used to visualize the areas of X-ray scans that the model was most sensitive to in order to verify that scans were classified based on features within the lungs.

# Dataset
The Kaggle Chest X-Ray Images (Pneumonia) dataset was used for this project, which can be found here: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. A total of 5840 images from the dataset were used (1580 healthy, 4260 pneumonia). Images were preprocessed by resizing to 160x160 pixels, applying a random rotation between -15 and 15 degrees, normalizing, and cropping around the edges of the images.

Example chest X-ray images shown in this repository are derived from the publicly available Kaggle dataset and are used solely for demonstration and interpretability analysis. The dataset itself is not included in this repository.

# Model
Architecture: Custom CNN trained from scratch
Loss Function: Binary Cross-Entropy with Logits 
Optimizer: Adam
Training Setup: 
* Train/val/test split: 70/15/15
* Model: Custom CNN
* Batch Size: 32
* Augmentation: Random rotation from -15 to 15 degrees
* Learning Rate: 5e-4
* Number of Epochs: 15

# Results

The following are the results from the model's evaluation on the test dataset.

Accuracy: 95%

ROC Curve:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/983a77b1-9980-43d5-880d-a6f3b0f0298e" />

AUC: 0.982

Confusion Matrix:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/26bbe620-bdc4-4828-9f28-94ce4d7dbb08" />

Activation Heatmap:

<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/032bbf6c-a2fc-420f-b5e0-e7d61af2f391" />

Note: Heatmaps use the 'VIRIDIS' colormap, with activation values increasing from red to yellow/green to blue.

* Sensitivity: 0.95
* Specificity: 0.93
* Positive Predictive Value: 0.97
* Negative Predictive Value: 0.87

# Limitations
The dataset used for this project is relatively small and is unbalanced between classes. Certain x-ray scans also include scanner artifacts that could potentially impact the model's learning. The model is only capable of performing binary classification and cannot distinguish between bacterial and viral pneumonia. This model is not clinically validated.

# How to run
* Download the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.
* Download this project
* Delete this project's empty 'archive' folder and replace it with the unzipped 'archive' folder you downloaded
* In Command Prompt or Power Shell: C:\...\Downloads\Pneumonia-X-Ray-Classifier-main\Pneumonia-X-Ray-Classifier-main> pip install -r requirements.txt
* C:\...\Downloads\Pneumonia-X-Ray-Classifier-main\Pneumonia-X-Ray-Classifier-main> python main.py


Repository Structure
+ archive/            # original dataset (not included)
+ models/             # trained models (contains my pretrained model)
+ results/            # stores figures from training and evaluation
+ cnn.py              # model architecture
+ dataset.py          # prepares dataset
+ train.py            # training script
+ eval.py             # Grad-CAM, confusion matrix, and ROC Curve scripts
+ main.py
+ requirements.txt
+ README.md

