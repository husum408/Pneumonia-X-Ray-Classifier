import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pathlib import Path
from os import listdir

#
# accuracy_fn
#
# Calculates the accuracy of the model for a single batch,
# assuming that logit values above 0.5 indicate pneumonia
# and that those less than or equal to 0.5 indicate a healthy
# xray scan.
#
# Args:
#   y_pred: logit values produced by the model for each image in a batch
#   y_true: label values for the batch
#
# Returns:
#   accuracy: proportion of correct predictions in the batch
#

def accuracy_fn(y_pred, y_true):
    y_pred = y_pred > 0.5 # Choose arbitrary logit threshold above which we
                          # classify images as "pneumonia". The choice of
                          # threshold does not affect training.
    y_pred[y_pred > 0.5] = 1
    num_correct = (y_pred == y_true).sum().item()
    accuracy = num_correct / len(y_true)
    return accuracy

#
# train_step
#
# Performs a single training step
#
# Args:
#   model: The model to be trained
#   dataloader: The dataset used for training
#   loss_fn: The loss function used to calculate model loss
#   optimizer: The optimizer used to update weights
#   device: The device where each image/label pair is sent.
#           This should match the device where the model was sent.
#
# Returns:
#   train_loss: The model's average loss across all batches in the dataloader
#   train_accuracy: The model's average accuracy across all batches in the dataloader
#

def train_step(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device) # send image/label pair to device
        y_pred_logits = model(X) # get model prediction

        loss = loss_fn(y_pred_logits.squeeze(), y.float()) # calculate loss across batch
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step() # update weights

        acc = accuracy_fn(y_pred=y_pred_logits.squeeze(), y_true=y) # calculate accuracy across batch
        train_acc += acc

    train_loss = train_loss / len(dataloader) # calculate average loss across training batches
    train_acc = train_acc / len(dataloader) # calculate average accuracy across training batches

    return train_loss, train_acc

#
# eval_step
#
# Performs a single evaluation step to further show the model's
# current performance. DOES NOT TRAIN THE MODEL. 
#
# Args:
#   model: The model to be evaluation
#   dataloader: The dataset used for evaluation
#   loss_fn: The loss function used to calculate model loss
#   device: The device where each image/label pair is sent.
#           This should match the device where the model was sent.
#
# Returns:
#   eval_loss: The model's average loss across all batches in the dataloader
#   eval_accuracy: The model's average accuracy across all batches in the dataloader
#

def eval_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device):
    
    model.eval()

    eval_loss, eval_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device) # send image/label pair to device
            eval_pred_logits = model(X) # get model prediction

            loss = loss_fn(eval_pred_logits.squeeze(), y.float()) # calculate loss across batch
            eval_loss += loss.item()

            acc = accuracy_fn(y_pred=eval_pred_logits.squeeze(), y_true=y) # calculate accuracy across batch
            eval_acc += acc

    eval_loss = eval_loss / len(dataloader) # calculate average loss across evaluation batches
    eval_acc = eval_acc / len(dataloader) # calculate average accuracy across evaluation batches

    return eval_loss, eval_acc

#
# loss_and_accuracy_curves
#
# Creates ad saves loss and accuracy curves for training
# and evaluation across all epochs.
#
# Args:
#   results: Dictionary of results with lists containing the train_loss,
#            train_acc, eval_loss, and eval_acc values for each epoch as
#            values for the 'train_loss', 'train_acc', 'eval_loss', and 'eval_acc' keys. 
#   num_epochs: The number of epochs used in training
#

def loss_and_accuracy_curves(results: dict, num_epochs: int, save_name: str):

    plt.figure(figsize=(15,7))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), results['train_loss'], label='train_loss')
    plt.plot(range(1, num_epochs + 1), results['eval_loss'], label='eval_loss')
    plt.title(save_name + ' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch Number')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), results['train_acc'], label='train_acc')
    plt.plot(range(1, num_epochs + 1), results['eval_acc'], label='eval_acc')
    plt.ylim((0, 1))
    plt.title(save_name + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch Number')
    plt.legend()

    save_folder_path = Path('results/' + save_name)

    if save_folder_path.is_dir() == False:
        save_folder_path.mkdir()

    existing_trials = len(listdir(save_folder_path))

    trial_num = 1

    if existing_trials != 0:
        trial_num = existing_trials + 1

    (save_folder_path / ('Trial' + str(trial_num))).mkdir()
    plt.savefig(str(save_folder_path) + '/Trial' + str(trial_num) + '/Loss_Accuracy_Curves')

#
# train
#
# Trains the model, plots the loss and accuracy curves
# and saves the trained model in the 'models' folder
#
# Args:
#   train_dataloader: Training dataset
#   val_dataloader: Validation dataset
#   model: Model to be trained
#   save_name: The trained model will be saved in the 'models' folder under this name
#   num_epochs: The number of epochs the model will be trained for
#   lr: The learning rate used for training
#

def train(train_dataloader, val_dataloader, model, save_name: str, num_epochs: int = 5, lr: float = 5e-4):

    # Use GPU for speed if available, otherwise use CPU

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    torch.manual_seed(42) # A manual seed was chosen for consistency but isn't required

    loss_fn = torch.nn.BCEWithLogitsLoss() 

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
    
    start_time = timer()

    results = {'train_loss': [],
        'train_acc': [],
        'eval_loss': [],
        'eval_acc': []}

    # Iterate through epochs, performing one train step and eval step for each epoch

    print('Training...\n')
    
    for epoch in range(0, num_epochs): 

        epoch_start_time = timer()

        print(f'Epoch: {epoch + 1}')

        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        
        eval_loss, eval_acc = eval_step(model=model,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        print(f'Train Loss: {train_loss}\nTrain Accuracy: {train_acc}\nEval Loss: {eval_loss}\nEval Accuracy: {eval_acc}\n')

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['eval_loss'].append(eval_loss)
        results['eval_acc'].append(eval_acc)

        epoch_end_time = timer()

        epoch_remaining_time = (epoch_end_time-epoch_start_time) * (num_epochs - 1 - epoch)
        epoch_remaining_min = epoch_remaining_time // 60
        epoch_remaining_sec = int(epoch_remaining_time % 60)

        print(f'Estimated time remaining: {epoch_remaining_min} minutes and {epoch_remaining_sec} seconds\n')

    end_time = timer()

    total_time = end_time - start_time
    total_time_min = total_time // 60
    total_time_sec = int(total_time % 60)

    print(f'Took {total_time_min} minutes and {total_time_sec} seconds\n')

    loss_and_accuracy_curves(results, num_epochs, save_name)

    # Todo: save loss and accuracy to a metrics file

    torch.save(model.state_dict(), 'models/' + save_name)
