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
from itertools import product
from collections import namedtuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.tensorboard import SummaryWriter

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

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

class CnnModel(nn.Module):

    def __init__(self, number_of_classes = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, number_of_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

def train_network(model, number_of_epoch, train_loader, val_loader, optimizer, loss_fn):

    tb = SummaryWriter(comment=comment)

    for epoch in range(number_of_epoch):

        true_labels = []
        predicted_labels = []
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        total = 0

        for batch in train_loader:

            drawings, labels = batch
            drawings = drawings.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(drawings)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total_acc += (labels == predicted).sum().item()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * total_acc / total
        val_total, val_total_acc, val_epoch_acc = validate_model(model, val_loader)


        tb.add_scalar('Loss', epoch_loss, epoch)
        tb.add_scalar('Correct', epoch_acc, epoch)


        print(" __Train dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%). Epoch loss: %.4f" % (total, total_acc, epoch_acc, epoch_loss))
        print("__Validation dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%)." % (val_total, val_total_acc, val_epoch_acc))

    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)

@torch.no_grad()
def validate_model(model, val_loader):

    total_acc = 0.0
    total = 0

    for batch in val_loader:

        drawings, labels = batch
        drawings = drawings.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        total += labels.size(0)

        outputs = model(drawings)

        _, predicted = torch.max(outputs.data, 1)

        total_acc += (labels == predicted).sum().item()

    epoch_acc = 100 * total_acc / total

    return total, total_acc, epoch_acc

class Run():
    @staticmethod
    def make_runs(parameters):
        run = namedtuple('run', parameters.keys())

        runs = []
        for x in product(*parameters.values()):
            runs.append(run(*x))

        return runs        



parm = dict(
    lr = [1e-5, 1e-4, 1e-3, 1e-2],
    opt = ['sgd', 'adam']
)




for run in Run.make_runs(parm):
    cnn_model = CnnModel(number_of_classes = 4)
    cnn_model = cnn_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    comment = f'-{run}'
    if run.opt == 'adam':
        optimizer = optim.Adam(cnn_model.parameters(), lr=run.lr, weight_decay=0.003)
    if run.opt == 'sgd':
        optimizer = optim.SGD(cnn_model.parameters(), lr=run.lr, weight_decay=0.0005)
    train_network(cnn_model, number_of_epoch = 40, train_loader = train_loader, val_loader = val_loader ,optimizer = optimizer, loss_fn = loss_fn)

