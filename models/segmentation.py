import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import bilinear
from torchvision.models import Wide_ResNet101_2_Weights, wide_resnet101_2


class Segmenter(nn.Module):
    # use a wideresnet101 backbone and inject coordinate features at the bottleneck

    def __init__(self, num_classes, use_coord = True, pretrained = True):
        super().__init__()
        self.use_coord = use_coord

        weights = Wide_ResNet101_2_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = wide_resnet101_2(weights=weights)

        bottleneck_in_ch = 1024 + (2 if use_coord else 0) #add 2 extra channels for coordinates if necessary

        # lightweight decoder at the end
        self.conv1 = nn.Conv2d(bottleneck_in_ch, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, in_tensor: torch.Tensor, coord: torch.Tensor = None):
        out = self.encoder.conv1(in_tensor)
        out = self.encoder.bn1(out)
        out = self.encoder.relu(out)
        out = self.encoder.maxpool(out)

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
