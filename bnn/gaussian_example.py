import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn.gaussian_step import GCLinear, gc_hardstep

# 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('using CUDA')
else:
    device = torch.device('cpu')
    print('WARNING: using CPU')

# Hyperparameters
input_size = 784  # 28x28
output_size = 10  # number of classes (digits) to predict
num_epochs = 3  # This is very low for demonstration purposes!
batch_size = 1
learning_rate = 0.001  # Adam default is 0.001
n_samples = 40 # number of samples for estimating the output variance

class GCNet(nn.Module):
    def __init__(self):
        super(GCNet, self).__init__()
        self.fc1 = GCLinear(784, 100)
        self.fc2 = GCLinear(100, 10)

    def forward(self, x):
        p, samples = x, x.repeat(n_samples, 1)
        #
        mu, sigma2, samples = self.fc1(x, samples)
        p, samples = gc_hardstep(mu, sigma2, samples)
        #
        mu, sigma2, samples = self.fc2(p, samples)
        # p, samples = gc_hardstep(mu, sigma2, samples)
        return mu
        # normalize the output probabilities
        # p += 1e-6 # we think of these as likelihoods for each class 
        # p /= p.sum()  # normalize to get posterior probs since uniform prior
        # return torch.log(p)  # CELoss expects logits (it does softmax internally)

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)
# take 10% of test and train


torch.manual_seed(0)  # set the random seed
model = GCNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dummy_data = torch.Tensor([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
])

dummy_labels = torch.Tensor([1, 2, 1, 2, 1, 2 ]) - 1


"""
for _ in range(100):
    for example, label in zip(dummy_data, dummy_labels):
        example = example.to(device)
        label = label.to(device)
        out = model(example)
        loss = criterion(out.reshape(1,-1), torch.tensor([int(label)]))
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
pass


"""
# Train the model
exp_rolling_loss = 2.5
alpha = 0.99
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        # TODO: handle batches
        images = images[0]
        labels = labels[0]
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # rolling loss
        exp_rolling_loss = alpha * exp_rolling_loss + (1 - alpha) * loss.item()
        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_loader), exp_rolling_loss))
        if (batch_idx+1) % 6000 == 0:
            # evaluate on the test set
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.view(-1, 28*28)
                    # batch
                    images = images[0]
                    labels = labels[0]
                    images = images.to(device)
                    labels = labels.to(device)
                    #
                    outputs = model(images)
                    predicted = torch.argmax(outputs)
                    total += 1
                    correct += (predicted == labels)
                print('Accuracy of the network on the 2000/10000 test images: {} %'.format(100 * correct / total))
        # """