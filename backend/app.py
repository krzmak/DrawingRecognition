from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from DrawingPreprocesing import align_to_top_left, scale_drawing, vector_to_raster, save_to_npy
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from model import CnnModel

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