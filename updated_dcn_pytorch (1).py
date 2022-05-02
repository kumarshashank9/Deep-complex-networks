import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from updated_clean_modules import *
import os
import numpy as np
import pdb
from pdb import set_trace as bp

on_zero=-1-1j
on_non_zero=+1+1j

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu")

def custom_replace(tensor, on_zero, on_non_zero):
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = Cardioid()

        self.spool = SpectralPooling2D(gamma=(0.5,0.5))
        self.avgpool = nn.AvgPool2d((32,32))

        self.complex_img = ComplexImg(3,3,(1,1),stride=1,padding='same')
        self.conv = ComplexConv(3,12,kernel_size=3,padding='same')
        self.bn = ComplexBatchNorm2d(12)

        self.residual_stage1r = ResidualBlock(12,12,kernel_size=3,shortcut='regular')
        self.residual_stage1p = ResidualBlock(12,12,kernel_size=3,shortcut='proj')

        self.residual_stage2r = ResidualBlock(24,24,kernel_size=3,shortcut='regular')
        self.residual_stage2p = ResidualBlock(24,24,kernel_size=3,shortcut='proj')

        self.residual_stage3r = ResidualBlock(48,48,kernel_size=3,shortcut='regular')

        self.out = nn.Linear(96,10)

    def forward(self, x):
        # Stage 1
        x = self.complex_img(x)
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
    
        # Stage 2
        for i in range(16):
            x = self.residual_stage1r(x)
            
        # Stage 3
        x = self.residual_stage1p(x)
        x = torch.fft.fft2(x)
        x = self.spool(x)
        x = torch.fft.ifft2(x)
        x = x.to(torch.float32)
        for i in range(15):
            x = self.residual_stage2r(x)

        # Stage 4
        x = self.residual_stage2p(x)
        x = torch.fft.fft2(x)
        x = self.spool(x)
        x = torch.fft.ifft2(x)
        x = x.to(torch.float32)
        for i in range(15):
            x = self.residual_stage3r(x)
        
        # Pooling
        x = torch.fft.fft2(x)
        x = self.spool(x)
        x = torch.fft.ifft2(x)
        x = x.to(torch.float32)
        x = self.avgpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x

    
net = Net()
#net = nn.DataParallel(net)
net= nn.DataParallel(net,device_ids = [1])

net = net.to(device)
summary(net,(3,32,32))

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,nesterov=True)
scheduler = CustomDCNScheduler(optimizer)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

#criterion = Complex_Loss()
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968,0.48215827,0.44653124), (0.24703233,0.24348505,0.26158768))
])
train_dataset = datasets.CIFAR10(
    'dataset', train=True, download=True, transform=transform)

test_dataset = datasets.CIFAR10(
    'dataset', train=False, download=True, transform=transform)

train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

min_valid_loss=np.inf

for epochs in range(50):
    scheduler.schedule(epochs)
    correct=0.0
    train_loss = 0.0
    net.train()
    for train_inputs, train_target in train_loader:
        train_inputs = train_inputs.to(device)
        train_target = train_target.to(device)
       
        train_out = net(train_inputs)
        loss = criterion(train_out,train_target)
        _, train_predicted = torch.max(train_out, 1)
        correct += (train_predicted == train_target).sum().item()
        
        net.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        train_loss +=loss.item()
       
    #########################################################################################Validation######################################
    net.eval()
    val_loss = 0.0
    correct_val = 0.0
    for val_inputs, val_target in val_loader:
        val_inputs = val_inputs.to(device)
        val_target = val_target.to(device)

        val_out = net(val_inputs)
        _, val_predicted = torch.max(val_out, 1)
        correct_val += (val_predicted == val_target).sum().item()
        valid_loss = criterion(val_out,val_target)
        val_loss +=valid_loss.item()
    
    if min_valid_loss > val_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        print()
        min_valid_loss = val_loss
         
        torch.save(net.state_dict(), 'dcn_best_model.pth')

    
    print('Training_Loss after epoch {:.2f} is {:.2f}'.format(epochs,train_loss / len(train_loader)))
    print()
    print('Validation_Loss after epoch {:.2f} is {:.2f}'.format(epochs,(val_loss / len(val_loader))))
    accuracy = correct/len(train_set) * 100.0
    val_acc = correct_val/len(val_set) *100.0
    print(f'{accuracy:.2f}% correct for train_set')
    print()
    print(f'{val_acc:.2f}% correct for val_set')
    torch.save(net.state_dict(), 'dcn_lastest_model.pth')


def test():
    correct = 0.0
    net = Net().to(device)
    net.load_state_dict(torch.load('dcn_lastest_model.pth'))
    net.eval()
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            out = net(inputs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == target).sum().item()
        accuracy = correct/len(test_dataset) * 100.0
        print(f'{accuracy:.2f}% correct on test_set')
    return
##############################################################################################

test()