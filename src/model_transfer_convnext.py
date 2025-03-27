import torch.nn as nn
import torchvision.models as models

class ConvNeXtAudioClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(ConvNeXtAudioClassifier, self).__init__()

        # Load pretrained ConvNeXt-Tiny
        self.base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        # Modify classifier head
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
