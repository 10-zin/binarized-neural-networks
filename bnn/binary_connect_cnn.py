import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn.binary_connect_fc import BCConv2d

import os

# Hyperparameters
input_size = 784  # 28x28
output_size = 10  # number of classes (digits) to predict
num_epochs = 40  # This is very low for demonstration purposes!
batch_size = 8
learning_rate = 0.002  # This is very high for demonstration purposes!
lr_decay_factor = 0.9  # exponentially decay the learning rate
model_path = './bnn/models'
model_basename = 'bnn_cnn'

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Define the ConvNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv2d = BCConv2d
        print("Using conv type: ", conv2d)
        self.conv1 = conv2d(1, 20, kernel_size=5)  # shape: (batch_size, 1, 28, 28) -> (batch_size, 10, 24, 24)
        self.conv2 = conv2d(20, 50, kernel_size=5)  # shape: (batch_size, 10, 12, 12) -> (batch_size, 20, 8, 8)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 64)  # 320, 50
        self.fc2 = nn.Linear(64, 10)   # 50, 10
        # print number of parameters
        print('Parameter count: ', sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2) / 2)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2) / 2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# data augmentation
transform_augmented = transforms.Compose([
    # transforms.RandomRotation(degrees=3), # Slight rotation
    transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3), # Translation, scaling, and shearing
    # transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness and contrast
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # Random erasing for robustness
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalization
])
trainset_augmented = datasets.MNIST(root='./data', train=True, transform=transform_augmented)
trainloader_augmented = DataLoader(dataset=trainset_augmented, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=False)


torch.manual_seed(0)  # set the random seed
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# evaluate on the test set
def evaluate():
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    return 100 * correct / total

# Train the model
def train():
    best_acc = 0
    save_path = None
    for epoch in range(num_epochs):
        model.train()  # set the model to training mode
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_factor
        for batch_idx, (images, labels) in enumerate(trainloader_augmented if epoch<20 else train_loader):
            torch.cuda.empty_cache()
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(trainloader_augmented), loss.item()))
        acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            if save_path is not None:
                os.remove(save_path)
            save_path = f'{model_path}/{model_basename}_epoch{epoch+1}_acc{acc:.2f}.pt'
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    # start time
    import time
    start = time.time()
    train()
    end = time.time()
    print(f'Time elapsed: {end - start}')

# EXPERIMENTS:
# full_precision 10x5²/20x5²/320/50 lr=0.002? decay=1.0 batch=64 epochs=10 dropout=0.5   (99.12% @ 35)
# binary         20x5²/25x5²/400/64 lr=0.002  decay=1.0 batch=8  epochs=10 dropout=0.5   97%
# binary         20x5²/25x5²/800/64 lr=0.003  decay=0.9 batch=8  epochs=10 dropout=0.5   DIVERGES
# binary         20x5²/50x5²/800/64 lr=0.001  decay=0.9 batch=8  epochs=10 dropout=0.5   96.3% (97% @ 16)
# binary         20x5²/50x5²/800/64 lr=0.002  decay=0.9 batch=8  epochs=10 dropout=0     97.15% (97.6% @ 15, 98.25% @ 36)
# binary         20x5²/50x5²/800/64 lr=0.002  decay=0.8 batch=8  epochs=10 dropout=0     11%@2, 
# binary         20x5²/50x5²/800/64 lr=0.001  decay=0.8 batch=8  epochs=10 dropout=0     45%@1, 94%
# binary, NO FC DROPOUT, DIV2 MAXPOOL  20x5²/50x5²/800/64 lr=0.002  decay=0.9 batch=8  epochs=10 dropout=0   (@1ablation: 92% -no_fc_dropout=88% -div2_maxpool=11.3%)
# binary, NO FC DROPOUT, DIV2 MAXPOOL  20x5²/50x5²/800/64 lr=0.002  decay=0.9 batch=8  epochs=10 dropout=0 98.27% (96.9%@3, 98%@8, 98.8%@18, 98.82%@28)
# binary, NO FC DROPOUT, DIV2 MAXPOOL, DATA_AUG  20x5²/50x5²/800/64 lr=0.002  decay=0.9 batch=8  epochs=10 dropout=0 97.68% (96.0%@3, 97.6%@8, 98.5%@18/39) 
# binary, NO FC DROPOUT, DIV2 MAXPOOL, DATA_AUG_MILD/2, NO_AUG@20  20x5²/50x5²/800/64 lr=0.002  decay=0.9 batch=8  epochs=10 dropout=0 98.5% (96.89%@3, 98.15%@5, 98.83%@17, 98.89%@27, 98.94%@32, 99.01%@37)