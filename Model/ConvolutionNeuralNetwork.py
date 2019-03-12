import torch.nn as nn
import torch.nn.functional as F

#Define the model for my neural network
#They should be a subclass Pytorch torch.nn.Module

class ConvolutionNeuralNetwork(nn.Module):
    
    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true)
        #3 channels in, 6 channels out, kernel 5
        self.conv1 = nn.Conv2d(3,6,5,5)
        
        #kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False
        #kernel is 2, stride is 2
        self.pool = nn.MaxPool2d(2,2)
        
        #6 channels in, 16 channels out, kernel 5
        self.conv2 = nn.Conv2d(6,16,5)
        
        #Linear 16 * 5 * 5 = 16 * 25 = 400 features in -> 120 out
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        #Linear 120 in -> 84 out
        self.fc2 = nn.Linear(120, 84)
        
        #Linear 84 in -> 10 out
        self.fc3 =  nn.Linear(84, 10)
        
    ## override method for Module
    # x is a tensor ... Matrix
    def forward(self, x):
        # Apply  to input of 3 channels conv1 (linear transforms) then relu (non-linear) and pool results
        # Pool applies to Cutoff values and Kernel size.
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        # adjust size
        x = x.view(-1, 16 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x
        
