import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class CnnModel(nn.Module):
    def __init__(self, number_of_classes=25):
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
        self.dropout = nn.Dropout(p=0.3)

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

classes = [
    'aircraft carrier', 'alarm clock', 'apple', 'basketball', 'bear', 'bee', 
    'bus', 'cake', 'carrot', 'cat', 'cup', 'dog', 'dragon', 'eye', 'flower', 
    'golf club', 'hand', 'house', 'moon', 'owl', 'pencil', 'pizza', 'shark', 
    'The Eiffel Tower', 'umbrella'
]

model = torch.load('model_v3.pth')
model.eval() 

mean = [0.1765]
std = [0.3540]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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

    print("Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{classes[i]}: {prob:.4f}")

    print(f"\nPredicted Class: {predicted_class}")

    plt.figure(figsize=(6, 6))
    plt.imshow(drawing, cmap="gray")
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis("off")
    plt.show()

classify(model, 'img9.npy', classes, transform, device)