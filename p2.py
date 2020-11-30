import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# The folloiwng lines to read data is refered from the internet
BATCH_SIZE = 512
EPOCH = 200
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True)

# Here's the architecture of NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.max_pool2d(output, 2, 2)
        output = F.relu(self.conv2(output))
        output = output.view(x.size(0), -1)
        output = F.relu(self.fc1(output))
        output = F.log_softmax(self.fc2(output), dim=1)
        return output

model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for i,(data,label) in enumerate(train_loader):
        data,label = data.to(device),label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if i == len(train_loader)-1:
            print('epoch ', epoch, 'complete!')
            with open('record.txt', 'a') as f:
                f.write('\t'.join([str(epoch), str(loss.item())]))
                f.write('\n')

def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,label in test_loader:
            data,label = data.to(device),label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,label,reduction='sum').item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test start! Loss: {:.4f}, Accuracy:{:.0f}%'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


for epoch in range(EPOCH):
    train(model,DEVICE,train_loader,optimizer,epoch)
print('Training Finished!')
test(model,DEVICE,test_loader)