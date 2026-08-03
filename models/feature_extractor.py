import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import List, Dict, Tuple

class FeatureExtractor(nn.Module):
    #wrapper for the backbone to extract and normalize the feature maps
    def __init__(self, backbone: nn.Module, layers_to_extract: List[str]):
        super().__init__()
        self.backbone = backbone
        self.layers_to_extract = layers_to_extract
        self.hooked_features = {}
        self._register_hooks()

    def _register_hooks(self):
        #registers all the forward hooks in the layers_to_extract list
        for name, module in self.backbone.named_modules():
            if name in self.layers_to_extract:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, layer_name):
        def hook(module, input_args, output):
            self.hooked_features[layer_name] = output
        return hook

    def _align_and_concatenate(self):
        #upsample all the hooked feature maps to whatever the highest spatial resolution is
        #and then concatenate them on the channel dimension
        aligned_features = []
        target_size = None

        for layer_name in self.layers_to_extract:
            #determine the largest spatial resolution
            feature_map = self.hooked_features[layer_name]
            if target_size is None or feature_map.shape[-2] > target_size[-2]:
                target_size = feature_map.shape[-2:]

        for layer_name in self.layers_to_extract:
            feature_map = self.hooked_features[layer_name]
            if feature_map.shape[-2:] != target_size:
                feature_map = functional.interpolate(
                    feature_map,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            aligned_features.append(feature_map)
        return torch.cat(aligned_features, dim=1)

    def forward(self, in_tensor: torch.Tensor):
        #pass the in_tensor to the backbone and process the hooked features
        self.hooked_features = {}

        with torch.no_grad():
            _ = self.backbone(in_tensor) #we just need to do a forward pass for the hooks to work

        processed_features = self._align_and_concatenate()

        #apply L2 normalization across the channel dimension
        normalized_features = functional.normalize(processed_features, p=2, dim=1)

        return normalized_features