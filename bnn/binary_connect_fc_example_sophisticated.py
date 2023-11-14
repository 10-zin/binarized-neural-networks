import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn.binary_connect_fc import BCLinear

import os

# Hyperparameters
input_size = 784  # 28x28
output_size = 10  # number of classes (digits) to predict
num_epochs = 40  # This is very low for demonstration purposes!
batch_size = 8
learning_rate = 0.007  # This is very high for demonstration purposes!
lr_decay_factor = 0.9  # exponentially decay the learning rate
layer_sizes = [input_size, 512, 256, 128, output_size]
use_dropout = False
dropout_prob = 0.2
model_path = './bnn/models'
model_basename = 'bnn_fc'

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        linear = BCLinear
        # linear = nn.Linear
        print('Using linear layer type:', linear)
        for k, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            setattr(self, f'fc{k+1}', linear(in_dim, out_dim))
            if use_dropout:
                setattr(self, f'dropout{k+1}', nn.Dropout(dropout_prob))
        # print num params
        print('Number of parameters: ', sum(p.numel() for p in self.parameters()))


    def forward(self, x):
        x = x.view(-1, 28*28)
        for k in range(1, len(layer_sizes)):
            x = getattr(self, f'fc{k}')(x)
            if k < len(layer_sizes) - 1:
                if use_dropout:
                    x = getattr(self, f'dropout{k}')(x)
                x = F.relu(x)
        return x

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

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
        for batch_idx, (images, labels) in enumerate(train_loader):
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))
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
# hidden=256, lr=0.005, epochs=15, batch_size=32, linear=BCLinear -> 96.75%
# hidden=1024/512/256,   lr=0.005 *=.2@%10, epochs=10, batch_size=32, linear=BCLinear -> 97.3% (97.9% @ 22)
# hidden=1024/512/256,   lr=0.005 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 98% (96.4% @ 3, 98.25% @ 21)
# hidden=1024/512/256,   lr=0.005 *=.2@%10, epochs=10, batch_size=1, linear=BCLinear -> (95.6% @ 3)
# hidden=1024/512/256,   lr=0.003 *=.4@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.8% (95.7% @ 3, 98% @ 22, 98.15% @ 27)
# hidden=1024/512/256,   lr=0.006 *=.4@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.7% (95.6% @ 3, 97.9% @ 12)
# hidden=1024/512/256,   lr=0.009 *=.4@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.5% (95.9% @ 3)

# hidden=1024/1024/1024, lr=0.005, epochs=10, batch_size=32, linear=BCLinear -> 97.2%
# hidden=512/256/128,    lr=0.005 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.5% (97.8% @ 21)
# hidden=2048,           lr=0.005 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 96.5%
# hidden=2048/1024/512,  lr=0.005 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.9% (98.25% @ 21)
# hidden=512/256/128,    lr=0.007 *=.9@%1,  epochs=10, batch_size=8, linear=BCLinear -> 97.3% (97.95% @ 25, 98.05% @ 39)
# hidden=512/256/128, Normalize(0.1307,0.3081)   lr=0.007 *=.9@%1,  epochs=10, batch_size=8, linear=BCLinear -> 97.2% (98.08% @ 35, 
# hidden=512/256/128, dropout=0.1, lr=0.007 *=.9@%1,  epochs=10, batch_size=8, linear=BCLinear -> 97.2% (96% @ 3, 98.15% @ 40)
# hidden=512/256/128, dropout=0.2, lr=0.007 *=.9@%1,  epochs=10, batch_size=8, linear=BCLinear -> (97.94% @ 38)

# hidden = 512/384/256,  lr=0.009 *=.2@%10, epochs=10, batch_size=32, linear=BCLinear -> 97.3% (97.8% @ 20)
# hidden = 512/384/256,  lr=0.009 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 96% (94.4% @ 3)
# hidden = 512/384/256,  lr=0.003 *=.2@%10, epochs=10, batch_size=8, linear=BCLinear -> 97.4% (95% @ 3, 97.8% @ 15)

# hidden = 128/128,      lr=0.005, epochs=10, batch_size=8, linear=BCLinear -> 97.2% (97.75% @ 21)

# CONCLUSIONS:
# -- small batch is essential for BCLinear because we shouldn't be using the same sampled weights. (This certainly helps with faster training, but perhaps not final accuracy.)
# -- a larger batch size can support a higher learning rate because the gradients are more stable.
# -- decaying the lr helps