import torch
import torch.nn as nn
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self, freeze=False):
        super(BaseModel, self).__init__()
        
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        # Freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
        num_features = self.model._modules['classifier']._modules['1'].in_features
        
        new_classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        num_features = self.model._modules['classifier']._modules['1'].in_features
        self.model._modules['classifier'] = new_classifier
        
    def forward(self, x):
        return self.model(x)
        