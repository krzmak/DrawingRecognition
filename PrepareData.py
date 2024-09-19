# Created by Krzysztof Mak for final engineering project at Politechnika Wrocławska

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json

with open('configuration.json', 'r') as config:
    location = json.load(config)

training_data_path = location.get("trainig data path")
validation_data_path = location.get("validation data path")
test_data_path = location.get("test data path")
raw_data_path = location.get("raw data path")


print(f"train data path: {training_data_path}")
print(f"val data path: {validation_data_path}")
print(f"test data path: {test_data_path}")
print(f"raw data path: {raw_data_path}")

def LoadNumpyFile(file):
    data = np.load(file)
    return data

def NumberOfDrawings(drawings):

    numberofdrawings =  np.size(drawings)//(1*784)
    return numberofdrawings

# data sets
# training data = 60 %
# validation data = 20 %
# test data = 20 % 

def SelectData(drawings):

    training_data = np.random.choice(drawings.shape[0], math.floor(NumberOfDrawings(drawings)*0.6), False)
    validation_data = np.random.choice(drawings.shape[0], math.floor(NumberOfDrawings(drawings)*0.2), False)
    test_data = np.random.choice(drawings.shape[0], math.floor(NumberOfDrawings(drawings)*0.2), False)

    return training_data, validation_data, test_data

def SaveData(drawings, train, val, test, train_path, val_path, test_path, file_name):
    train_output = drawings[train]
    val_output = drawings[val]
    test_output = drawings[test]
    file_name_without_ext = os.path.splitext(file_name)[0]
    np.save(os.path.join(train_path, file_name_without_ext + '_train.npy'), train_output)
    np.save(os.path.join(val_path, file_name_without_ext + '_val.npy'), val_output)
    np.save(os.path.join(test_path, file_name_without_ext + '_test.npy'), test_output)

def SeparateFiles(input_directory):
    directory = os.fsencode(input_directory)
    
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        file_path = input_directory + '/' + file_name
        drawings = LoadNumpyFile(file_path)
        numberofdrawings = NumberOfDrawings(drawings)
        data = SelectData(drawings)
        train = data[0]
        val = data[1]
        test = data[2]
        SaveData(drawings, train, val, test, training_data_path, validation_data_path, test_data_path, file_name)

        print("file name:", file_name)
        print("number of images in train data:", np.size(train))
        print("number of images in val data:", np.size(val)) 
        print("number of images in test data:", np.size(test)) 
        print("number of images in this file:", numberofdrawings) 

SeparateFiles(raw_data_path)