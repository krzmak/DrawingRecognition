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

# Załaduj model
model = torch.load('model_v3.pth')
model.eval()  # Przełączenie modelu w tryb oceny

mean = [0.1765]
std = [0.3540]

#experimental func

def classify_image(model, image_path, classes, mean, std, device=None):
    # Determine the device (CPU or CUDA)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the appropriate device
    model.to(device)

    # Load the image as a numpy array
    image = np.load(image_path)

    # Transform the image to match the model's expected input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Apply the transform to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Send the image to the model and make predictions
    with torch.no_grad():
        output = model(image_tensor)

    # Get the probabilities for each class
    probabilities = F.softmax(output, dim=1)
    probabilities = probabilities.cpu().numpy().squeeze()  # Move to CPU and convert to numpy array

    # Get the predicted class
    predicted_class = classes[np.argmax(probabilities)]

    # Display the probabilities for all classes
    print("Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{classes[i]}: {prob:.4f}")

    print(f"\nPredicted Class: {predicted_class}")

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis("off")
    plt.show()

# Call the function to classify the saved image
classify_image(model, 'image.npy', classes, mean, std)
