from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
import os
import shutil
import random
import matplotlib.pyplot as plt

#
# shuffle_images
#
# Shuffles a list of images and reorganizes them into new lists 
# following a 70:15:15 ratio between the amount of images allocated to the training 
# set list, test set list, and validation set list. This function was used since the 
# dataset downloaded from Kaggle has very few images in the validation set (only 18 total).
#
# Args:
#   image_path_list: a list containing all of the images for one class
#
# Returns:
#   train_path_list: a new list of images to be used as the training set
#   test_path_list: a new list of images to be used as the test set
#   val_path_list: a new list of images to be used as the validation set
#

def shuffle_images(image_path_list: list):

    random.seed(42)

    random.shuffle(image_path_list)

    n = len(image_path_list) // 20

    train_path_list = image_path_list[:(14*n)]
    test_path_list = image_path_list[(14*n):(17*n)]
    val_path_list = image_path_list[(17*n):(20*n)]

    return train_path_list, test_path_list, val_path_list

#
# custom_crop
#
# Crops out the top 40 pixels of an image and 20 pixels from both the 
# left and right sides of the image. Implemented as part of the 
# data transform to combat consistent edge activation around the top 
# and sides of the images.
#
# Args:
#   image: Image tensor to be cropped
# 
# Returns:
#   image: Image tensor following cropping 
#

def custom_crop(image: Tensor):

    image = image[:, 40:, 20:-20]

    return image

# 
# apply_data_transform
#
# Transforms train, test, and validation data using one data transform by
# resizing, applying random rotation, converting to tensors, cropping, and
# normalizing images. 
#
# Args:
#   train_path: location of the folder containing training image data
#   test_path: location of the folder containing test image data
#   val_path: location of the folder containing validation image data
#

def apply_data_transform(train_path: str, test_path: str, val_path: str):

    data_transform = transforms.Compose([
        transforms.Resize(size=(200, 200)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Lambda(custom_crop),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    train_data = datasets.ImageFolder(root=train_path,
                                      transform=data_transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=test_path,
                                     transform=data_transform,
                                     target_transform=None)

    val_data = datasets.ImageFolder(root=val_path,
                                    transform=data_transform,
                                    target_transform=None)

    return train_data, test_data, val_data

#
# make_dataloaders
#
# Creates dataloaders for training, test, and validation data.
#
# Args:
#   train_data: Transformed training data
#   test_data: Transformed test data
#   val_data: Transformed validation data
#   num_workers: Number of subprocesses to use for loading data in parallel. 
#                Defaults to 0 to prevent freezing but can be adjusted for 
#                better speed.
#
# Returns:
#   train_dataloader: Shuffled training dataloader
#   test_dataloader: Test dataloader
#   val_dataloader: Validation dataloader
#   val_dataloader_for_eval: Validation dataloader with batch size one. Use
#                            later to evaluate model.
#

def make_dataloaders(train_data: datasets.ImageFolder, test_data: datasets.ImageFolder, val_data: datasets.ImageFolder, num_workers: int = 0):

    NUM_WORKERS = num_workers

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=32,
                                num_workers=NUM_WORKERS,
                                shuffle=True)

    test_dataloader = DataLoader(dataset = test_data,
                                batch_size=1,
                                num_workers=NUM_WORKERS,
                                shuffle=False)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=32,
                                num_workers=NUM_WORKERS,
                                shuffle=False)
    
    val_dataloader_for_eval = DataLoader(dataset=val_data,
                                         batch_size=1,
                                         num_workers=NUM_WORKERS,
                                         shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader, val_dataloader_for_eval

# 
# preprocess
# 
# Reorganizes and processes the dataset to prepare for model learning.
#
# Args:
#   data_path: Location of the original dataset
#
# Returns:
#   train_dataloader: Shuffled training dataloader
#   test_dataloader: Test dataloader
#   val_dataloader: Validation dataloader
#

def prepare_dataset(data_path: Path = Path('archive/chest_xray/chest_xray')):

    # Check that data_path location exists and print its contents

    if data_path.is_dir():

        print('\nOriginal Data:')

        for dpath, dnames, filenames in os.walk(data_path):
            print(f'{dpath}: {len(dnames)} directories and {len(filenames)} files')

        print()

    else:
        raise Exception(f'Failed to load {data_path}')


    # Check to see if the 'reorganized_data' folder containing reorganized data already exists. 
    # If not, reorganize the original data, storing it in the 'reorganized_data' folder. This 
    # was done since the dataset downloaded from Kaggle has very few images in the validation set.

    new_data_path = Path('reorganized_data')

    if (new_data_path.is_dir() == False):

        new_data_path.mkdir()

        # make lists containing the paths of every image for both classes and shuffle 
        # the paths into new train, test, and validation sets (split 70:15:15).

        normal_image_path_list = list(data_path.glob('*/NORMAL/*.jpeg'))
        pneumonia_image_path_list = list(data_path.glob('*/PNEUMONIA/*.jpeg'))


        [normal_train_path_list, normal_test_path_list, normal_val_path_list] = shuffle_images(normal_image_path_list)
        [pneumonia_train_path_list, pneumonia_test_path_list, pneumonia_val_path_list] = shuffle_images(pneumonia_image_path_list)

        # Make 'train', 'test', and 'val' subfolders containing a subfolder for each class
        # in the 'reorganized_data' folder. Then, copy the images from the original data 
        # into these new subfolders according to the shuffled path lists. 

        for set_type in ['train', 'test', 'val']:
            (new_data_path / set_type / 'NORMAL').mkdir(parents=True, exist_ok=True)
            (new_data_path / set_type / 'PNEUMONIA').mkdir(parents=True, exist_ok=True)

        path_lists = [normal_train_path_list,
                      normal_test_path_list,
                      normal_val_path_list,
                      pneumonia_train_path_list,
                      pneumonia_test_path_list,
                      pneumonia_val_path_list]
                      
        locations = ['train/NORMAL', 'test/NORMAL', 'val/NORMAL', 'train/PNEUMONIA', 'test/PNEUMONIA', 'val/PNEUMONIA']

        for i in range(len(path_lists)):
            for image in path_lists[i]:
                shutil.copy(image, new_data_path / locations[i])


    # Print the contents of the 'reorganized_data' folder

    print('Reorganized Data:')
    for dpath, dnames, filenames in os.walk(new_data_path):
        print(f'{dpath}: {len(dnames)} directories and {len(filenames)} files')
    print()

    # Apply transforms to all of the images and load the datasets into dataloaders
    # to be used by the model

    train_path = new_data_path / 'train'
    test_path = new_data_path / 'test'
    val_path = new_data_path / 'val'

    train_data, test_data, val_data = apply_data_transform(train_path, test_path, val_path)
    train_dataloader, test_dataloader, val_dataloader, val_dataloader_for_eval = make_dataloaders(train_data, test_data, val_data)

    return train_dataloader, test_dataloader, val_dataloader, val_dataloader_for_eval
