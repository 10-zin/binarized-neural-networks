import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn.binary_connect_fc import BCLinear


# Hyperparameters
input_size = 784  # 28x28
output_size = 10  # number of classes (digits) to predict
num_epochs = 3  # this is very low for demonstration purposes
batch_size = 100
learning_rate = 0.02  # This is very high for demonstration purposes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BCLinear(input_size, 100)
        self.fc2 = BCLinear(100, 100)
        self.fc3 = BCLinear(100, output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(0)  # set the random seed
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))