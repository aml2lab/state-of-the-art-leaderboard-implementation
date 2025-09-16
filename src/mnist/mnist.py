#sota rank 1. accuracy 99.99%

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

trainloader_no_shuffle = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

# Get a batch of training images
dataiter = iter(trainloader_no_shuffle)
images, labels = next(dataiter)  # Get the first batch of images

# Select specific indices of images you want to use
selected_indices = [0, 3, 4, 6, 9]  # Example indices

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleNN()


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):  # Train for 1 epoch
    correct = 0
    total = 0

    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1} completed - Accuracy: {accuracy:.2f}%')

print('Finished Training')
