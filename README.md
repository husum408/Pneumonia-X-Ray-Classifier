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
Loss Function: Binary Cross Entropy with Logits (pos_weight = 0.6)
Optimizer: Stochastic Gradient Descent
Training Setup: 
* Train/val/test split: 70/15/15
* Model: Custom CNN
* Batch Size: 32
* Augmentation: Random rotation from -15 to 15 degrees

# Results
Test Accuracy: 88%
ROC Curve:

<img width="568" height="422" alt="image" src="https://github.com/user-attachments/assets/d0fbace4-a5e7-4383-82ae-c6d14292c38f" />

AUC: 0.956

Confusion Matrix:

<img width="555" height="415" alt="image" src="https://github.com/user-attachments/assets/4b20e468-32b5-4f82-b1b0-9fbde536ddbf" />

Activation Heatmap:

<img width="520" height="531" alt="image" src="https://github.com/user-attachments/assets/f0251ec1-1a9f-43aa-abee-e84471566c75" />

Sensitivity: 0.87
Specificity: 0.91
Positive Predictive Value: 0.96
Negative Predictive Value: 0.73

# Limitations
The dataset used for this project is relatively small and is unbalanced between classes. Certain x-ray scans also include scanner artifacts that could potentially impact the model's learning. The model is only capable of performing binary classification and cannot distinguish between bacterial and viral pneumonia. This model is not clinically validated.

# How to run
* Download the dataset from here: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.
* Replace this project's empty 'archive' folder with the unzipped 'archive' folder you downloaded
* pip install -r requirements.txt
* python main.py


