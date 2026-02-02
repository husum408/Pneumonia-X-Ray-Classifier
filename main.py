from dataset import prepare_dataset
from cnn import CNN
from train import train
from eval import evaluate
from torch import load
from pathlib import Path

def main():

    train_dataloader, test_dataloader, val_dataloader, val_dataloader_for_eval = prepare_dataset()

    # The model's parameters can be adjusted here. Only models trained using
    # the same parameters as the one's you are currently using can be loaded properly

    model = CNN(input_shape=3, 
                hidden_units=8,
                output_shape=1,
                kernel_size=7,
                padding=3,
                image_size_x=160,
                image_size_y=160)
    
    model_name = 'pneumonia_xray_model' # Set the name of your model. If it already exists, 
                                        # it will be loaded. If not, a new model will be trained
                                        # and saved in the 'models' folder under this name.

    if Path('models/' + model_name).is_file() == True:

        model.load_state_dict(load('models/' + model_name, weights_only = False))

    else:

        lr_schedule = [1e-2, 1e-2] # Set your desired learning rate schedule for training (simple example provided)

        train(train_dataloader, val_dataloader, lr_schedule, model, model_name)


    final_evaluation = True # When ready for the final evaluation set this to true

    if final_evaluation == True:
        evaluate(model, test_dataloader, model_name)
    else:
        evaluate(model, val_dataloader_for_eval, model_name)

if __name__ == "__main__":
    main()
