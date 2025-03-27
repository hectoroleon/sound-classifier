import torch.nn as nn
import torchvision.models as models

class ResNetAudioClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(ResNetAudioClassifier, self).__init__()

        # Load pretrained ResNet18
        self.base_model = models.resnet18(pretrained=True)

        # Adapt input channels (from 3 → 3 but we'll duplicate in dataset)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
