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

dat = LoadNumpyFile(training_data_path + '/' + 'full_numpy_bitmap_airplane_train.npy')
print(np.shape(dat[1]))
