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

with open('configuration.json', 'r') as config:
    path = json.load(config)

training_data_path = path.get("trainig data path")
validation_data_path = path.get("validation data path")
test_data_path = path.get("test data path")
raw_data_path = path.get("raw data path")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



cnn_model = CnnModel()
network_oper = NetworkOperations(cnn_model, check_point_name= 'model7_checkpoint.pth.tar') #if you waant new model change to model 8
cnn_model = cnn_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2, weight_decay=0.0005)

network_oper.train_network(number_of_epoch = 20, train_loader = train_loader, val_loader = val_loader ,optimizer = optimizer, loss_fn = loss_fn, csv_save_path= 'data7.csv') #if you waant new model change to data 8

checkpoint = torch.load('model7_checkpoint.pth.tar') #if you waant new model change to model 8

cnn_model = CnnModel()
cnn_model.load_state_dict(checkpoint['model'])
cnn_model = cnn_model.to(device)

torch.save(cnn_model, 'model_v7.pth') #if you waant new model change to model 8