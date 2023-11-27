# https://nextjournal.com/gkoehler/pytorch-mnist

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Turn on for determinism.
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def make_loader(train, batch_size):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            './files/',
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        ),
        batch_size=batch_size,
        shuffle=train,
    )

train_loader = make_loader(True, batch_size_train)
test_loader = make_loader(False, batch_size_test)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = func.relu(func.max_pool2d(self.conv1(x), 2))
        x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, training=self.training)
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)

network = Net()
network.to(device)
optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,
    momentum=momentum,
)

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = func.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} Loss: {:.6f}'.format(
        epoch,
        loss.item(),
    ))
    #torch.save(network.state_dict(), './results/model.pth')
    #torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += func.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
    ))

try:
    network.load_state_dict(torch.load('./mnist.model'))
    test()
except:
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

torch.save(network.state_dict(), './mnist.model')
