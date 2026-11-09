import torch
import torch.nn as nn
from torchvision.models import (
    resnet101, ResNet101_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    wide_resnet50_2, Wide_ResNet50_2_Weights,
)
import clip
from dataclasses import dataclass
from typing import Callable

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class ResNetTrunk(nn.Module):
    #Stem + layer1 through 4, runs only up to deepest stage needed
    def __init__(self, stem: nn.Module, stages: list, max_stage: int):
        super().__init__()
        self.stem = stem
        self.max_stage = max_stage
        for i, stage in enumerate(stages[:max_stage], start=1):
            setattr(self,f"layer{i}", stage) #direct children, so hook names are "layer2" etc.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval() #basic freezing of parameters

    def train(self, mode: bool = True):
        return super().train(False) #basically NEVER let it get put in train mode

    def forward(self, x):
        x = self.stem(x)
        for i in range(1, self.max_stage+1):
            x=getattr(self,f"layer{i}")(x)
        return x

def _from_torchvision(m, max_stage):
    stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
    return ResNetTrunk(stem, [m.layer1, m.layer2, m.layer3, m.layer4], max_stage)

def _clip_builder(name):
    #we're leveraging DESIGN PATTERNS in this now >:D
    #it's factory closure world forever and ever maybe
    def build(max_stage):
        model, _ = clip.load(name, device="cpu") #start out on cpu, it's the caller's job to move it
        v = model.visual.float() #only keep the visual encoder, we don't need the text encoder
        stem = nn.Sequential(v.conv1, v.bn1, v.relu1, v.conv2, v.bn2, v.relu2,
                             v.conv3, v.bn3, v.relu3, v.avgpool)
        return ResNetTrunk(stem, [v.layer1, v.layer2, v.layer3, v.layer4], max_stage)
    return build

def _tv_builder(ctor, weights):
    return lambda max_stage: _from_torchvision(ctor(weights=weights), max_stage)

def _dino_rn50(max_stage):
    m = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    return _from_torchvision(m, max_stage)

@dataclass(frozen=True)
class BackboneSpec: #this just lets us save the individual backbone configurations easily ad reuse them later
    build: Callable
    mean: tuple
    std: tuple

BACKBONES = {
    #dictionary of different backbones
    #CLIP family
    "clip_rn101":  BackboneSpec(_clip_builder("RN101"), *CLIP_STATS),
    "clip_rn50":   BackboneSpec(_clip_builder("RN50"), *CLIP_STATS),
    "clip_rn50x4": BackboneSpec(_clip_builder("RN50x4"), *CLIP_STATS),
    # Matched-architecture controls (isolate the effect of pretraining)
    "resnet101_in1k_v2": BackboneSpec(_tv_builder(resnet101, ResNet101_Weights.IMAGENET1K_V2), *IMAGENET_STATS),
    "wrn101_in1k_v2":    BackboneSpec(_tv_builder(wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2), *IMAGENET_STATS),
    # Comparability with PatchCore-style papers
    "wrn50_in1k_v1": BackboneSpec(_tv_builder(wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V1), *IMAGENET_STATS),
    # Self-supervised control
    "dino_rn50": BackboneSpec(_dino_rn50, *IMAGENET_STATS),
}

ALIASES = {"CLIP": "clip_rn101", "WResNet101": "wrn101_in1k_v2"} #might as well have this because of previous tests

def _resolve(name: str) -> str:
    name = ALIASES.get(name, name)
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {sorted(BACKBONES)}")
    return name

def get_norm_stats(name: str):
    #this is going to be useful for the dataloaders
    spec = BACKBONES[_resolve(name)]
    return spec.mean, spec.std

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



def init_backbone(name: str, layers: list) -> nn.Module:
    bad = [l for l in layers if l not in ("layer1", "layer2", "layer3", "layer4")] #just checking if theres an errant layer
    if bad:
        raise ValueError(f"Unsupported layers: {bad}")
    max_stage = max(int(l.removeprefix("layer")) for l in layers)
    return BACKBONES[_resolve(name)].build(max_stage)