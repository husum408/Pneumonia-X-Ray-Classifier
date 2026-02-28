from torch import inference_mode, mean, max, manual_seed
from torch.cuda import is_available
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from os import listdir

#
# confusion_matrix
#
# Plots a confusion matrix
# 
# Args:
#   predictions: List of predictions logits produced by the model for each image in a dataset
#   labels: List of labels for each image in a dataset
#   threshold: Logit threshold above which predictions indicate pneumonia
#   eval_folder_path: Location to save figure
#

def confusion_matrix(predictions: list, labels: list, threshold, eval_folder_path):

    num_true_positive = 0
    num_false_positive = 0
    num_true_negative = 0
    num_false_negative = 0
    
    # For each prediction, determine if it is a true positive, 
    # false positive, true negative, or false negative

    for i in range(len(predictions)):
        if predictions[i] > threshold: 
            if labels[i] == 1:
                num_true_positive += 1
            elif labels[i] == 0:
                num_false_positive += 1
        else:
            if labels[i] == 1:
                num_false_negative += 1
            elif labels[i] == 0:
                num_true_negative += 1
    
    # Plot confusion matrix
    
    results = [[num_true_negative, num_false_negative], [num_false_positive, num_true_positive]]

    fig, ax = plt.subplots()
    ax.matshow(results)
    
    for (i, j), k in np.ndenumerate(results):
        ax.text(j, i, k, ha='center', va='center')
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predictions')
    ax.set_xticks(range(2), ['NORMAL', 'PNEUMONIA'])
    ax.set_yticks(range(2), ['NORMAL', 'PNEUMONIA'])
    ax.xaxis.set_ticks_position('bottom')
    
    plt.savefig(eval_folder_path + '/confusion_matrix')
    
    # Calculate and print other statistics

    accuracy = (num_true_positive+num_true_negative)/len(predictions)
    sensitivity = (num_true_positive)/(num_true_positive+num_false_negative)
    specificity = (num_true_negative)/(num_false_positive+num_true_negative)
    ppv = (num_true_positive)/(num_true_positive+num_false_positive)
    npv = (num_true_negative)/(num_true_negative+num_false_negative)
    
    print('Confusion Matrix Stats:')
    print(f'Accuracy: {accuracy: 0.2f}')
    print(f'Sensitivity: {sensitivity: 0.2f}')
    print(f'Specificity: {specificity: 0.2f}')
    print(f'Positive Predictive Value: {ppv: 0.2f}')
    print(f'Negative Predictive Value: {npv: 0.2f}\n')

    # Todo: Save metrics to file

    return

#
# plot_roc_curve
#
# Plots an ROC curve and determines the threshold that
# provides the optimal balance between sensitivity and specificity.
#
# Args:
#   predictions: List of predictions logits produced by the model for each image in a dataset
#   labels: List of labels for each image in a dataset
#   eval_folder_path: Location to save figure
#
# Returns:
#   optimal_threshold: The threshold corresponding to the point on the 
#                      ROC curve closest to (0, 1). This threshold optimizes
#                      the balance between sensitivity and specificity.
#

def plot_roc_curve(predictions, labels, eval_folder_path):

    fpr, tpr, thresholds = roc_curve(labels, predictions)

    roc_auc = auc(fpr, tpr)

    # Determine the index that corresponds to the point closest
    # to (0, 1)

    dist = 2
    fpr_index = 0

    for i in range(len(fpr)):
        new_dist = np.sqrt((fpr[i])**2 + (1 - tpr[i])**2)
        if dist > new_dist:
            dist = new_dist
            fpr_index = i
    
    optimal_threshold = thresholds[fpr_index]
    threshold_fpr = fpr[fpr_index]
    threshold_tpr = tpr[fpr_index]

    fig = plt.figure()
    plt.title('ROC Curve')
    plt.plot(fpr, tpr)
    plt.scatter(threshold_fpr, threshold_tpr, c='red', label='Threshold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    plt.savefig(eval_folder_path + '/roc_curve')
    
    print('ROC Curve Stats:')
    print(f'The area under the curve is: {roc_auc: 0.3f}')
    print(f'The false positive rate at the threshold is: {threshold_fpr: 0.3f}')
    print(f'The true positive rate at the threshold is: {threshold_tpr: 0.3f}')
    print(f'The optimal threshold is: {optimal_threshold: 0.3f}\n')

    # Todo: Save metrics to file

    return optimal_threshold

#
# grad_cam
#
# Plots a heatmap showing the model's activations across an image
# 
# Args:
#   model: The model who's activations will be shown
#   img: The image to be analyzed by the model, preferrably taken from a dataloader.
#   eval_folder_path: Location to save figure
#    


def grad_cam(model, img, eval_folder_path):

    model.eval()

    pred = model(img)
    
    # get the model's activations using hooks and multiply 
    # by pooled gradients to make the heatmap

    pred.backward()
    
    gradients = model.get_activations_gradient()
    
    pooled_gradients = mean(gradients, dim=[2, 3])
    
    activations = model.get_activations(img).detach()
    
    for i in range(2):
        activations[:, i, :, :] *= pooled_gradients[:, i]
    
    heatmap = mean(activations, dim=1).squeeze()
    
    heatmap = np.maximum(heatmap, 0)
    
    heatmap /= max(heatmap)
    
    numpy_img = img.numpy().squeeze().transpose([1, 2, 0])
    
    # Extrapolate the heatmap to the size of the original image and superimpose it

    heatmap_new = cv2.resize(heatmap.numpy(), (numpy_img.shape[1], numpy_img.shape[0]))
    
    heatmap_new = np.uint8(255 * heatmap_new)
    
    heatmap_new = cv2.applyColorMap(heatmap_new, cv2.COLORMAP_VIRIDIS)
    
    superimposed_img = heatmap_new * 0.0025 + ((numpy_img + 1) / 2) 
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Activation Heatmap')
    plt.imshow(heatmap, cmap='viridis')
    
    plt.subplot(1, 3, 2)
    plt.title('Original Image')
    plt.imshow((numpy_img + 1)/ 2 )
    
    plt.subplot(1, 3, 3)
    plt.title('Superimposed Image')
    plt.imshow(np.clip(superimposed_img, 0, 1))
    
    plt.tight_layout()

    plt.savefig(eval_folder_path + '/grad_cam')

    return

#
# evaluate
#
# Performs a single evaluations step, then plots the confusion matrix,
# ROC curve, and Grad-CAM
#
# Args:
#   model: The model to be evaluated
#   dataloader: The dataset on which the model's performance will
#               be evaluated.
#   model_name: Name of the model to be evaluated
#

def evaluate(model, dataloader, model_name):

    # Find if the last trial for the given model has an 'eval' folder.
    # If not, only training has been done. Create the eval folder for the trial
    # and save there. If it exists, evaluation for that trial has already been done.
    # Make a new trial folder and make an eval subfolder to save the new figures as a new trial.

    save_folder_path = Path('results/' + model_name)

    if save_folder_path.is_dir() == False:
        save_folder_path.mkdir(parents=True)

    existing_trials = len(listdir(save_folder_path))

    if existing_trials != 0:
        if Path(save_folder_path / ('Trial' + str(existing_trials)) / 'eval').is_dir():
            trial_num = existing_trials + 1
        else:
            trial_num = existing_trials
    else:
        trial_num = 1

    eval_folder_path = str(save_folder_path) + '/Trial' + str(trial_num) + '/eval'
    Path(eval_folder_path).mkdir(parents=True)


    # Check if GPU is available

    device = 'cuda' if is_available() else 'cpu'
    
    # Run evaluation step while keeping track of prediction logits and corresponding labels

    manual_seed(42) # This is implemented for consistency but can be removed

    print(f'Evaluation of {model_name}:\n')

    predictions = []
    labels = []

    model.eval()

    with inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions.append(pred.item())
            labels.append(y.item())

    img, label = next(iter(dataloader))

    optimal_threshold = plot_roc_curve(predictions, labels, eval_folder_path) 

    threshold = optimal_threshold # The logit threshold above which images are classified
                                  # as "pneumonia" can be adjusted. Decreasing it from 
                                  # the optimal threshold will improve sensitivity at the 
                                  # cost of specificity, while increasing it will have the
                                  # opposite effect.

    confusion_matrix(predictions, labels, threshold, eval_folder_path)

    grad_cam(model, img, eval_folder_path)

    return

    