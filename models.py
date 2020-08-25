import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class ModifiedAlexNet(nn.Module):
    def __init__(self):
        super(ModifiedAlexNet, self).__init__()
        model = models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 3)
        self.alexnet = model

    def forward(self, x):

        return self.alexnet(x)

"""
1. Add your model class. 
2. The name of your feature extraction module must be "features".
3. A pretrained model must be prepared.

class custom(nn.Moduele):
    def __init__(self):
        super(AlexNet, self).__init__()
        .....
        .....
"""

class ModelFactory():
    def __init__(self):
        pass

    def load_model(self, modeltype):

        if modeltype == 'alexnet':
            # Pretrained model on ImageNet dataset
            model = models.alexnet(pretrained=True)

        elif modeltype == 'modified_alexnet':
            # Example
            model = ModifiedAlexNet()
            model.load_state_dict(torch.load('./pretrained_publishinghouse.pth'))

        elif modeltype == 'custom':
            pass
            # Declare your model class
        
        elif modeltype == 'vgg16':   
            model = models.vgg16(pretrained=True)

        else:
            print("Please check the model name again.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(model)

        # model freezer
        for param in model.parameters():
            param.requires_grad = False

        model = model.to(device)
        print(device, 'operation is available.')

        return model
