import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
import json

from data_prep import NumpyDrawingsDataset

with open('configuration.json', 'r') as config:
    path = json.load(config)

training_data_path = path.get("trainig data path")
validation_data_path = path.get("validation data path")
test_data_path = path.get("test data path")
raw_data_path = path.get("raw data path")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

training_dataset = NumpyDrawingsDataset(training_data_path, transform= data_transforms)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True)

def calculate_mean_and_std(loader):
    """
        x
    """
    mean = 0.
    std = 0.
    total_number_of_drawings = 0

    for drawings, _ in loader:
        number_of_drawings_in_batch = drawings.size(0)
        drawings = drawings.view(number_of_drawings_in_batch, drawings.size(1), -1)
        mean += drawings.mean(2).sum(0)
        std += drawings.std(2).sum(0)
        total_number_of_drawings += number_of_drawings_in_batch

    mean /= total_number_of_drawings
    std /= total_number_of_drawings


    return mean, std

def show_images_of_given_label(dataset, label):
    batch_size = 64
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    img = []
    for i in range(batch_size):
        if labels[i] == label:
            img.append(images[i])
    grid = torchvision.utils.make_grid(img, nrow = 5)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()
 
print(calculate_mean_and_std(training_loader))