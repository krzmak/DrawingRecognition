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

with open('test_configuration.json', 'r') as config:
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

mean = [0.1971]
std = [0.3692]

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean),torch.tensor(std))
])

training_dataset = NumpyDrawingsDataset(training_data_path, transform= data_transforms)
validation_dataset = NumpyDrawingsDataset(validation_data_path, transform= data_transforms)
testing_dataset = NumpyDrawingsDataset(test_data_path, transform= data_transforms)

print(training_dataset.data[2])
print(np.shape(training_dataset.data[1]))


"""class CnnModel(nn.Module):

    def __init__(self, number_of_classes = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):

    """

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
    
show_images_of_given_label(training_dataset, 1)