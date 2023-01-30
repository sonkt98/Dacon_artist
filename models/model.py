import math
import torch.nn as nn
import torchvision.models as models
import timm
from models.tiny_vit import tiny_vit_5m_224, tiny_vit_21m_224, tiny_vit_21m_384


class BaseModel(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class TinyVit5m224(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = tiny_vit_5m_224(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class TinyVit21m224(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = tiny_vit_21m_224(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class TinyVit21m384(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = tiny_vit_21m_384(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class TinyVit21m384Cls(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = tiny_vit_21m_384(pretrained=True)
        classifier = nn.Linear(576, num_classes)
        nn.init.xavier_uniform_(classifier.weight)
        stdv = 1. / math.sqrt(classifier.weight.size(1))
        classifier.bias.data.uniform_(-stdv, stdv)
        self.backbone.head = classifier

    def forward(self, x):
        x = self.backbone(x)
        return x


class EfficientNetB4NS(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        classifier = nn.Linear(1792, num_classes)
        nn.init.xavier_uniform_(classifier.weight)
        stdv = 1. / math.sqrt(classifier.weight.size(1))
        classifier.bias.data.uniform_(-stdv, stdv)
        self.backbone.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        return x
