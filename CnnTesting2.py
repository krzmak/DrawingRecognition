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

#new imports
from model import CnnModel
from data_prep import NumpyDrawingsDataset
from model import NetworkOperations

from itertools import product
from collections import namedtuple

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

training_dataset = NumpyDrawingsDataset(training_data_path, transform=data_transforms)
validation_dataset = NumpyDrawingsDataset(validation_data_path, transform=data_transforms)
testing_dataset = NumpyDrawingsDataset(test_data_path, transform=data_transforms)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=256, shuffle=True)

parm = dict(
    lr=[1e-5, 1e-4, 1e-3, 1e-2],
    opt=['sgd', 'adam']
)

os.makedirs("results", exist_ok=True)

model_counter = 12  # Licznik modeli i checkpointów, początek od 12

class Run():
    @staticmethod
    def make_runs(parameters):
        run = namedtuple('run', parameters.keys())

        runs = []
        for x in product(*parameters.values()):
            runs.append(run(*x))

        return runs    

for run in Run.make_runs(parm):

    cnn_model = CnnModel(number_of_classes=25)
    cnn_model = cnn_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    if run.opt == 'adam':
        optimizer = optim.Adam(cnn_model.parameters(), lr=run.lr, weight_decay=0.003)
    elif run.opt == 'sgd':
        optimizer = optim.SGD(cnn_model.parameters(), lr=run.lr, weight_decay=0.0005)

    checkpoint_name = f"model{model_counter}_checkpoint.pth.tar"
    csv_file_name = f"data{model_counter}.csv"
    model_file_name = f"model_v{model_counter}.pth"
    
    network_oper = NetworkOperations(cnn_model, check_point_name=checkpoint_name)

    network_oper.train_network(
        number_of_epoch=20,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        csv_save_path=os.path.join("results", csv_file_name)
    )
    
    checkpoint = torch.load(checkpoint_name)
    cnn_model.load_state_dict(checkpoint['model'])
    cnn_model = cnn_model.to(device)
    torch.save(cnn_model, os.path.join("results", model_file_name))

    model_counter += 1