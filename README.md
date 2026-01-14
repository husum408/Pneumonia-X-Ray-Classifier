# Pneumonia Detection for Chest X-Rays Using CNNs
This project uses a convolutional neural network for binary classification of chest X-ray scans as healthy or indicating pneumonia. Using a publicly available dataset, I trained and evaluated a classifier and analyzed the model's behavior using tools such as Grad-CAM.

# Motivation
Classifiers are useful in medical imaging as tools for diagnosing patients. A healthy patient's lungs will appear dark in a chest x-ray, while a patient with pneumonia's lungs will show white patches due to fluid buildup. This difference can be readily analyzed using computer vision models.

Grad-CAM was used to visualize the areas of X-ray scans that the model was most sensitive to in order to verify that scans were classified based on features within the lungs.

# Dataset
The Kaggle Chest X-Ray Images (Pneumonia) dataset was used for this project, which can be found here: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. A total of 5840 images from the dataset were used (1580 healthy, 4260 pneumonia). Images were preprocessed by resizing to 128x128, applying a random rotation between -30 and 30 degrees, normalizing, and cropping the top 20 pixels of the image.

# Model
Architecture: Custom CNN trained from scratch
Loss Function: Binary Cross Entropy with Logits
Optimizer: Stochastic Gradient Descent
Training Setup:

# Results
Test Accuracy: 
ROC-AUC Curve:

Confusion Matrix:

Calibration:


Sensitivity:
Specificity:
Positive Predictive Value:
Negative Predictive Value:

# Limitations
The dataset used for this project is relatively small and is unbalanced between classes. Certain x-ray scans also include scanner artifacts that could potentially impact the model's learning. The model is only capable of performing binary classification and cannot distinguish between bacterial and viral pneumonia. This model is not clinically validated.

# How to run



