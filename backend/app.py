from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
from DrawingPreprocesing import align_to_top_left, scale_drawing, vector_to_raster, save_to_npy
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class CnnModel(nn.Module):
    def __init__(self, number_of_classes = 25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, number_of_classes)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x

classes = [
    'aircraft carrier', 'alarm clock', 'apple', 'basketball', 'bear', 'bee', 
    'bus', 'cake', 'carrot', 'cat', 'cup', 'dog', 'dragon', 'eye', 'flower', 
    'golf club', 'hand', 'house', 'moon', 'owl', 'pencil', 'pizza', 'shark', 
    'The Eiffel Tower', 'umbrella'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnModel()
model = torch.load('model_v10.pth', map_location=device)
model.to(device)
model.eval()


transform = transforms.Compose([
        transforms.ToTensor(),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify(model, drawing, classes, transform, device):

    model.to(device)

    drawing = np.load(drawing)

    #byc moze do edycji    
    drawing = drawing.squeeze()

    drawing_tensor = transform(drawing).unsqueeze(0)

    drawing_tensor = drawing_tensor.to(device)

    with torch.no_grad():
        output = model(drawing_tensor)
    
    probabilities = F.softmax(output, dim=1)
    probabilities = probabilities.cpu().numpy().squeeze()

    predicted_class = classes[np.argmax(probabilities)]

    class_probabilities = {classes[i]: float(prob) for i, prob in enumerate(probabilities)}

    return predicted_class, class_probabilities   

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS to allow frontend-backend communication

@app.route('/')
def serve_index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/classify-drawing', methods=['POST'])
def classify_drawing():
    drawing_data = request.get_json()
    
    # Preprocess and classify as before
    aligned = align_to_top_left(drawing_data)
    scaled_drawing = scale_drawing(aligned)
    raster_img = vector_to_raster([scaled_drawing])
    output_file = 'temp_drawing.npy'
    save_to_npy(raster_img, output_file)

    predicted_class, class_probabilities = classify(model, output_file, classes, transform, device)

    print({
    'predicted_class': predicted_class,
    'class_probabilities': class_probabilities
    })

    return jsonify({
        'predicted_class': predicted_class,
        'class_probabilities': class_probabilities
    })


if __name__ == '__main__':
    app.run(debug=True)