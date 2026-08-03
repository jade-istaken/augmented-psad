import torch
import torch.nn as nn
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights

def init_backbone() -> nn.Module:
    #initializes the resnet backbone with pre-trained imagenet weights
    weights = Wide_ResNet101_2_Weights.IMAGENET1K_V2
    model = wide_resnet101_2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False #freeze the weights to preserve imagenet performance

    model.eval() # set to eval mode in order to disable updating

    return model