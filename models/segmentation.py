import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import bilinear
from torchvision.models import Wide_ResNet101_2_Weights, wide_resnet101_2
from models import init_backbone
import clip

class Segmenter(nn.Module):
    # use a wideresnet101 backbone and inject coordinate features at the bottleneck

    def __init__(self, num_classes, use_coord = True, pretrained = True):
        super().__init__()
        self.use_coord = use_coord
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model, _ = clip.load("RN101", device=device)
        self.encoder = clip_model.visual
        self.encoder = self.encoder.float()

        for param in self.encoder.parameters():
            param.requires_grad = False

        bottleneck_in_ch = 1024 + (2 if use_coord else 0) #add 2 extra channels for coordinates if necessary

        # lightweight decoder at the end
        self.conv1 = nn.Conv2d(bottleneck_in_ch, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, in_tensor: torch.Tensor, coord: torch.Tensor = None):
        x = in_tensor.float()

        out = self.encoder.relu1(self.encoder.bn1(self.encoder.conv1(x)))
        out = self.encoder.relu2(self.encoder.bn2(self.encoder.conv2(out)))
        out = self.encoder.relu3(self.encoder.bn3(self.encoder.conv3(out)))
        out = self.encoder.avgpool(out)

        features_0 = self.encoder.layer1(out) # skip connection 1 (high res)
        features_1 = self.encoder.layer2(features_0) #skip connection 2 (mid res)
        out = self.encoder.layer3(features_1) #bottleneck

        #coordinate integration
        if self.use_coord and coord is not None:
            coord_interpolation = functional.interpolate(
                coord, size=out.shape[-2:], mode="bilinear", align_corners=True
            )
            out = torch.cat([out,coord_interpolation], dim=1)

        #lightweight decoder
        out = functional.interpolate(out, size=features_1.shape[-2:], mode="bilinear", align_corners=True)
        out = self.relu(self.conv1(out)) + features_1

        out = functional.interpolate(out, size=features_0.shape[-2:], mode="bilinear",align_corners=True)
        out = self.relu(self.conv2(out)) + features_0

        out = functional.interpolate(out, size=in_tensor.shape[-2:], mode="bilinear", align_corners=True)
        out = self.conv3(out)

        return out
