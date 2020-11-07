import torch
import torchvision
import torch.nn as nn
# ResNet architecture for training
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet=torchvision.models.resnet101(pretrained=True)
        self.fc1=torch.nn.Linear(1000,512)
        self.fc2=torch.nn.Linear(512,48)
    def forward(self,image):
        out=self.resnet(image)  
        out=self.fc1(out)
        out=self.fc2(out)
        return out
