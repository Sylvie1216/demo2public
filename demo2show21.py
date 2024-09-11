import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
#Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
Y = lfw_people.target
print("X_min:",X.min(),"X_train_max:",X.max())
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
# reshape (batch_size, channels, height, width)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:",X_train.shape)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DATA
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
# CNN
class SimpleCNN(nn.Module):
    #initialize
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        #The size of the flattening after the convolution and pooling layers, calculated dynamically during the first forward propagation
        self.flattened_size = None
        # FUlly
        self.fc1 = nn.Linear(1, 128)  # 将会在 forward 中动态更新
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #Reduce the space size
        x = self.pool(x)
        #The pooled feature map is flattened into a one-dimensional vector
        x = x.view(x.size(0), -1)

        if self.flattened_size is None:
            #calculate the features' number
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 128)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
num_classes = len(lfw_people.target_names)
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
# use Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
# TRAINING
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            #calucate margin Cross entropy loss function
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')
# Testing proceedure
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    #Reduce memory footprint
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
train_model(model, train_loader, criterion, optimizer, epochs=10)
test_model(model, test_loader)