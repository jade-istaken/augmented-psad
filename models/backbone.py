import torch
import torch.nn as nn
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
import clip


class CLIPResNetBackbone(nn.Module):
    # initializes the CLIP backbone with pre-trained weights
    def __init__(self, model_name: str = "RN101"):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load(model_name, device=device)
        self.visual = self.clip_model.visual
        self.visual = self.visual.float()

        self.layer1 = self.visual.layer1
        self.layer2 = self.visual.layer2
        self.layer3 = self.visual.layer3
        self.layer4 = self.visual.layer4
        # these just make it easier to interface with the rest of the codebase

        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()  # set to eval mode in order to disable updating

    def forward(self, x: torch.Tensor):
        # skip layer 4 because there was a spatial mismatch because of attention pooling
        x = x.type(self.visual.conv1.weight.dtype)
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def init_backbone() -> nn.Module:
    return CLIPResNetBackbone()