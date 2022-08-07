
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
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
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
                       transforms.ToTensor()
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=1

        second_layer_size = 10

        # 1st layer
        self.fc1 = BinarizeLinear(28*28,second_layer_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(second_layer_size)

        # 2nd layer
        self.fc2 = BinarizeLinear(second_layer_size, second_layer_size)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(second_layer_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.relu(x)

        return x

model = Net()
if args.cuda:
    torch.cuda.set_device(3)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def sign(var):
    if var >= 0:
        return 1
    else:
        return 0

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        data = data.sign()
        #print("data_bin: ",data)

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

    fc1_weights = model.state_dict()['fc1.weight']
    fc1_bias = model.state_dict()['fc1.bias']
    fc2_weights = model.state_dict()['fc2.weight']
    fc2_bias = model.state_dict()['fc2.bias']

    # bn1_weights = model.state_dict()['bn1.weight']
    # bn1_bias = model.state_dict()['bn1.bias']
    print(fc1_weights.shape)

    second_layer_size = 10

    import sys

    sys.stdout = open('stdout123.txt', 'w')
    print(model)
    #print(model.state_dict())

    print("fc1: ========================================================")
    bin_weights = Binarize(fc1_weights)
    bin_weights = torch.transpose(bin_weights,0, 1)
    #print("bin_weights: ",bin_weights)
    #print("fc1: ========================================================")
    #print("weights: ", bin_weights)
    for i in range(784):
        print("    {", end='')
        for j in range(second_layer_size):
            val = int(bin_weights[i][j].item())

            if(j == 783):
                print(str(val), end='')
            else:
                print(str(val)+", ", end='')
        print("},")
    print("\n--------------------------------------------------------------------------------")
    print("fc1_bias: ",fc1_bias.shape)
    bin_bias = Binarize(fc1_bias)
    for i in range(second_layer_size):
        #val = round(fc1_bias[i].item(), 6)
        val = int(bin_bias[i].item())
        print(str(val)+", ", end='')
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
        #print(str(val) + "f" + ", ", end='')
        #print("\n--------------------------------------------------------------------------------")
        val = round(math.sqrt(1/(bn1_running_var[i].item()+0.00001)), 6)
        print(str(val)+"f"+", ", end='')





    print("\n fc2: ========================================================")
    bin_weights2 = Binarize(fc2_weights)
    bin_weights2 = torch.transpose(bin_weights2,0, 1)
    #print("bin_weights: ",bin_weights)
    #print("fc1: ========================================================")
    #print("weights: ", bin_weights)
    for i in range(second_layer_size):
        print("    {", end='')
        for j in range(second_layer_size):
            val = int(bin_weights2[i][j].item())

            if(j == second_layer_size-1):
                print(str(val), end='')
            else:
                print(str(val)+", ", end='')
        print("},")
    print("\n--------------------------------------------------------------------------------")
    print("fc2_bias: ",fc2_bias.shape)
    bin_bias2 = Binarize(fc2_bias)
    for i in range(second_layer_size):
        val = int(bin_bias2[i].item())
        print(str(val)+", ", end='')
    print("\n--------------------------------------------------------------------------------")

    print("bn2: ========================================================")

    print("========================================================")
    bn2_running_mean = model.state_dict()['bn2.running_mean']
    print("bn2.running_mean: ", bn1_running_mean.shape)
    for i in range(second_layer_size):
        val = round(bn2_running_mean[i].item(), 6)
        print(str(val)+"f"+", ", end='')

    print("\n--------------------------------------------------------------------------------")

    import math
    bn2_running_var = model.state_dict()['bn2.running_var']
    print("bn2.running_var_modified: ", bn2_running_var.shape)
    for i in range(second_layer_size):
        #print(str(val) + "f" + ", ", end='')
        #print("\n--------------------------------------------------------------------------------")
        val = round(math.sqrt(1/(bn2_running_var[i].item()+0.00001)), 6)
        print(str(val)+"f"+", ", end='')

    model.eval()
    model.parameters()
    test_loss = 0
    correct = 0

    print("\n--------------------------------------------------------------------------------")

    flag = 0
    torch.set_printoptions(profile="full")
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.sign()
            #print("data_bin: ", data)

            print("data.shape: ",data.shape)
            printed_data = data

            if flag==0:
                print("data: ", printed_data.view(20,784))
                print("target: ",target)
                flag = 1
            data.shape: torch.Size([20, 1, 28, 28])



            print("target.shape: ", target.shape)
            print("target: ", target)

            output = model(data)

            test_loss += criterion(output, target).item() # sum up batch loss
            print("output.data.shape: ", output.data.shape)
            print("output.data: ",output.data)
            print("output.data.max: ", output.data.max(1, keepdim=True))
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            print("pred: ", pred.shape)
            if flag==0:
                print("pred: ", pred)
            print("target.data: ", target.data)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            print("-------------------------------------------------------------------------------")

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
