import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as transforms
import torch

from Model.ConvolutionNeuralNetwork import ConvolutionNeuralNetwork
from Model.LettersDataset import LettersDataset

class Training():
    
    def __init__(self):
        self.net = ConvolutionNeuralNetwork();
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = LettersDataset("/Users/sameriksson/temp/handwriting", transform, True)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
        testset = LettersDataset("/Users/sameriksson/temp/handwriting", transform, False)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=2)
        
    def train(self):
        for epoch in range(100):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(data[inputs])

                loss = self.criterion(outputs, data[labels])
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0

        print('Finished Training')
    
    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.net(data[images])
                _, predicted = torch.torch.max(outputs.data, 1)
                total += data[labels].size(0)
                correct += (predicted == data[labels]).sum().item()

                print('Accuracy of the network on the %d test images: %d %%' % (
                    total, 100 * correct / total))