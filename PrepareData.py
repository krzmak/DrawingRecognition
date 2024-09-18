# Created by Krzysztof Mak for final engineering project at Politechnika Wrocławska

import numpy as np
import math
import matplotlib.pyplot as plt

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

def SaveData(drawings, train, file_name):
    output = drawings[train]
    np.save(file_name, output)


file = 'cnn_data/data_raw/full_numpy_bitmap_cat.npy'
file_save = 'test.npy'

drawings = LoadNumpyFile(file)
numberofdrawings = NumberOfDrawings(drawings)

data = SelectData(drawings)

train = data[0]
val = data[1]
test = data[2]

SaveData(drawings, train, file_save)

print(train)

print("number of images in train data:", np.size(train))
# print("number of images in train data:", numberofdrawings(val)) 
# print("number of images in train data:", numberofdrawings(test)) 
print("number of images in this file:", numberofdrawings) 
