import torch
import torch.nn as nn
import torchvision.models as models
import core.config.configuration as cnfg

class InceptionClassifier(nn.Module):
    def __init__(self):
        super(InceptionClassifier, self).__init__()
        # Load a pre-trained Inception v1 (GoogLeNet)
        self.inception = models.googlenet(pretrained=True)

        for param in self.inception.parameters():
            param.requires_grad = False

        self.num_ftrs = self.inception.fc.in_features

        self.inception.fc = nn.Sequential(
            nn.Linear(self.num_ftrs, 150),
            nn.BatchNorm1d(150),            
            nn.ReLU(),                       
            nn.Dropout(0.5),               
            nn.Linear(150, cnfg.num_classes),            
        )

    def forward(self, x):
        # Forward pass through the pre-trained Inception v1 layers
        x = self.inception(x)
        return x
    
class VGG16Classifier(nn.Module):
    def __init__(self):
        super(VGG16Classifier, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze the parameters so we don't backprop through them
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])

        dummy_input = torch.randn(1, cnfg.channel_in, cnfg.image_width, cnfg.image_width)  
        dummy_features = self.vgg16(dummy_input)
        self.feature_size = dummy_features.view(-1).shape[0]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 150),  
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(150, cnfg.num_classes)  
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x



class ResClassifier(nn.Module):
    def __init__(self):
        super(ResClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Identity()

        # Forward a dummy variable through the feature extractor to determine input size
        dummy_input = torch.randn(1, cnfg.channel_in, cnfg.image_width, cnfg.image_width)
        dummy_features = self.resnet(dummy_input)
        self.feature_size = dummy_features.view(-1).shape[0]
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.feature_size, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),              
            nn.Dropout(0.5),          
            nn.Linear(150, cnfg.num_classes),         
        )

    def forward(self, x):
        # Forward pass through the pre-trained VGG16 layers
        x = self.resnet(x)
        return x