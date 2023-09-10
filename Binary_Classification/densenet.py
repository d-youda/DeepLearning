import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = models.densenet121(pretrained=True)
        num_ft = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Sequential(
            nn.Linear(num_ft, 32),
            nn.ReLU(inplace=True),            
            nn.Linear(32,1),
            nn.Softmax(dim=1)
        )
    
    def forward(self,x):
        output = self.cnn(x)
        return output