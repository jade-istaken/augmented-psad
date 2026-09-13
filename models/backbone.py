import torch
import torch.nn as nn
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
import clip

def init_backbone() -> nn.Module:
    #initializes the CLIP backbone with pre-trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load('RN101', device=device)

    visual_encoder = model.visual

    for param in visual_encoder.parameters():
        param.requires_grad = False #freeze the weights to preserve CLIP performance

    visual_encoder.eval() # set to eval mode in order to disable updating

    return model