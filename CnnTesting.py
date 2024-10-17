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
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

#mean = [0.1971]
#std = [0.3692]

mean = [0.1765]
std = [0.3540]


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean),torch.tensor(std))
])

training_dataset = NumpyDrawingsDataset(training_data_path, transform= data_transforms)
validation_dataset = NumpyDrawingsDataset(validation_data_path, transform= data_transforms)
testing_dataset = NumpyDrawingsDataset(test_data_path, transform= data_transforms)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=256, shuffle=True)

class CnnModel(nn.Module):

    def __init__(self, number_of_classes = 25):
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
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x 

def train_network(model, number_of_epoch, train_loader, val_loader, optimizer, loss_fn):

    max_acc = 0
    
    with open('data2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Epoch', 'Train Total', 'Train Correct', 'Train Accuracy', 'Loss', 
                         'Validation Total', 'Validation Correct', 'Validation Accuracy'])

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

            val_total, val_total_acc, val_epoch_acc, val_confiusion_matrix = validate_model(model, val_loader)

            if val_epoch_acc > max_acc:
                max_acc = val_epoch_acc
                save_checkpoint(model, epoch, optimizer, max_acc)

            writer.writerow([epoch + 1, total, total_acc, epoch_acc, epoch_loss,
                             val_total, val_total_acc, val_epoch_acc])

            print(" __Train dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%). Epoch loss: %.4f" % (total, total_acc, epoch_acc, epoch_loss))
            print("__Validation dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%)." % (val_total, val_total_acc, val_epoch_acc))

        cm = confusion_matrix(true_labels, predicted_labels)

        print(cm)

        print(val_confiusion_matrix)


@torch.no_grad()
def validate_model(model, val_loader):

    true_labels = []
    predicted_labels = []

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

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

    epoch_acc = 100 * total_acc / total

    cm = confusion_matrix(true_labels, predicted_labels)

    return total, total_acc, epoch_acc, cm


def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'model2_checkpoint.pth.tar')


cnn_model = CnnModel()
cnn_model = cnn_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4, weight_decay=0.003)

train_network(cnn_model, number_of_epoch = 100, train_loader = train_loader, val_loader = val_loader ,optimizer = optimizer, loss_fn = loss_fn)

checkpoint = torch.load('model2_checkpoint.pth.tar')

cnn_model = CnnModel()
cnn_model.load_state_dict(checkpoint['model'])
cnn_model = cnn_model.to(device)

torch.save(cnn_model, 'model_v2.pth')