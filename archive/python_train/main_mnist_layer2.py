from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=1

        second_layer_size = 200

        # 1st layer
        self.fc1 = BinarizeLinear(28*28,second_layer_size)
        self.bn1 = nn.BatchNorm1d(second_layer_size)
        self.relu1 = nn.ReLU()

        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.hardsigmoid(x)


        return x

model = Net()
if args.cuda:
    torch.cuda.set_device(3)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    print(model)
    fc1_weights = model.state_dict()['fc1.weight']
    fc1_bias = model.state_dict()['fc1.bias']
    bn1_weights = model.state_dict()['bn1.weight']
    bn1_bias = model.state_dict()['bn1.bias']
    print(fc1_weights.shape)

    second_layer_size = 200

    import sys

    sys.stdout = open('stdout.txt', 'w')

    print("fc1: ========================================================")

    for i in range(second_layer_size):
        print("{", end='')
        for j in range(784):
            val = round(fc1_weights[i][j].item(), 6)
            if(j == 783):
                print(str(val)+"f", end='')
            else:
                print(str(val)+"f"+", ", end='')
        print("},")
    print("\n--------------------------------------------------------------------------------")
    print("fc1_bias: ",fc1_bias.shape)
    for i in range(second_layer_size):
        val = round(fc1_bias[i].item(), 6)
        print(str(val)+"f"+", ", end='')
    print("\n--------------------------------------------------------------------------------")


    print("bn1: ========================================================")


    print("========================================================")
    bn1_running_mean = model.state_dict()['bn1.running_mean']
    print("bn1.running_mean: ", bn1_running_mean.shape)
    for i in range(second_layer_size):
        val = round(bn1_running_mean[i].item(), 6)
        print(str(val)+"f"+", ", end='')

    print("\n--------------------------------------------------------------------------------")

    import math
    bn1_running_var = model.state_dict()['bn1.running_var']
    print("bn1.running_var_modified: ", bn1_running_var.shape)
    for i in range(second_layer_size):
        val = round(math.sqrt(1/(bn1_running_var[i].item()+0.00001)), 6)
        print(str(val)+"f"+", ", end='')

    model.eval()
    model.parameters()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)

            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    sys.stdout.close()



# for epoch in range(1, args.epochs + 1):
for epoch in range(1, 2):
    train(epoch)
    #train(1)
    test()
    if epoch%40==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
