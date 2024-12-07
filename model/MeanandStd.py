import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import math
import os
import json

with open('configuration.json', 'r') as config:
    path = json.load(config)

training_data_path = path.get("trainig data path")
validation_data_path = path.get("validation data path")
test_data_path = path.get("test data path")
raw_data_path = path.get("raw data path")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyDrawingsDataset(Dataset):
    """Numpy drawing dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        self.npy_files = [f for f in os.listdir(root_dir)]
        self.labels = {npy_file: i for i, npy_file in enumerate(self.npy_files)}

        self.data = []
        self.data_labels = []

        for npy_file in self.npy_files:
            file_path = root_dir + '/' + npy_file
            data = np.load(file_path)
            self.data.append(data)
            label = self.labels[npy_file] 
            self.data_labels.append(np.full((data.shape[0],), label))

        self.data = np.concatenate(self.data, axis=0)
        self.data_labels = np.concatenate(self.data_labels, axis=0)        

    def __len__(self):
        """Returns the size of dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """ Support the indexing such that dataset[i] can be used to get i(th) sample"""
        data = self.data[idx]
        data_labels = self.data_labels[idx]

        data = data.reshape(64,64)

        if self.transform:
            data = self.transform(data)

        return data, data_labels

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