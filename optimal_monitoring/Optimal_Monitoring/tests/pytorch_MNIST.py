import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

dim = 60

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


net = ConvNet()
if torch.cuda.is_available():
    net.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_function = nn.CrossEntropyLoss()

train_dataset = MNIST('./data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((dim, dim)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))
test_dataset = MNIST('./data', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.Resize((dim, dim)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ]))

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=8)

epochs = 10
steps = 0
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    net.train()
    for images, labels in train_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        log_ps = net(images)
        loss = loss_function(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                log_ps = net(images)
                test_loss += loss_function(log_ps, labels)
                top_p, top_class = log_ps.topk(1, dim=1)
                equals = top_class.flatten().long() == labels
                accuracy += torch.mean(equals.float()).item()
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        print("[Epoch: {}/{}] ".format(e+1, epochs),
              "[Training Loss: {:.3f}] ".format(running_loss/len(train_loader)),
              "[Test Loss: {:.3f}] ".format(test_loss/len(test_loader)),
              "[Test Accuracy: {:.3f}]".format(accuracy/len(test_loader)))