import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data import Mnist
from model import LeNet5

# Generate training set
train_set = Mnist(
    root=r'C:\Users\young\PycharmProjects\pythonProject\LeNet-5\MNIST_data',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])
)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True
)

# Instantiate a network
net = LeNet5()

# Defining loss functions and optimizers
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)

# 3 Training model
loss_list = []
for epoch in range(10):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, start=0):

        images, labels = data  # Read the data of a batch
        optimizer.zero_grad()  # Gradient reset, initialization
        outputs = net(images)  # Forward propagation
        loss = loss_function(outputs, labels)  # calculation error
        loss.backward()  # Back propagation
        optimizer.step()  # Weight update
        running_loss += loss.item()  # Error accumulation

        # Print the loss value every 300 batch
        if batch_idx % 300 == 299:
            print('epoch:{} batch_idx:{} loss:{}'
                  .format(epoch + 1, batch_idx + 1, running_loss / 300))
            loss_list.append(running_loss / 300)
            running_loss = 0.0  # Error clearing

print('Finished Training.')

# Print loss value change curve
import matplotlib.pyplot as plt

plt.plot(loss_list)
plt.title('traning loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# test
test_set = Mnist(
    root=r'C:\Users\young\PycharmProjects\pythonProject\LeNet-5\MNIST_data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=32,
    shuffle=True
)

correct = 0  # Predicted correct number
total = 0  # Total pictures

for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()

print('Test set accuracy: {}%'.format(100 * correct // total))

# Test handwritten digits designed by yourself
from PIL import Image

I = Image.open('8.jpg')
L = I.convert('L')
plt.imshow(L, cmap='gray')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])

im = transform(L)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    _, predict = torch.max(outputs.data, 1)
    print(predict)
