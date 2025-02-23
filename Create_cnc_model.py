import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import json

#new imports
from model import CnnModel
from data_prep import NumpyDrawingsDataset
from model import NetworkOperations

with open('test_configuration.json', 'r') as config:
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
validation_dataset = NumpyDrawingsDataset(validation_data_path, transform= data_transforms)
testing_dataset = NumpyDrawingsDataset(test_data_path, transform= data_transforms)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=256, shuffle=True)



cnn_model = CnnModel()
network_oper = NetworkOperations(cnn_model, check_point_name= 'model_checkpoint.pth.tar')  
cnn_model = cnn_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4, weight_decay=0.003)

network_oper.train_network(number_of_epoch = 20, train_loader = train_loader, val_loader = val_loader ,optimizer = optimizer, loss_fn = loss_fn, csv_save_path= 'data.csv') #save training data loss_fc,accuracy etc. to csv file

checkpoint = torch.load('model_checkpoint.pth.tar') #

cnn_model = CnnModel()
cnn_model.load_state_dict(checkpoint['model'])
cnn_model = cnn_model.to(device)

torch.save(cnn_model, 'model.pth')