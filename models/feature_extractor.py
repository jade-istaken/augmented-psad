import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import List, Dict, Tuple

class FeatureExtractor(nn.Module):
    #wrapper for the backbone to extract and normalize the feature maps
    def __init__(self, backbone: nn.Module, layers_to_extract: List[str],norm_mode="joint"):
        super().__init__()
        assert norm_mode in ("joint", "layer_pixel", "layer_pooled")
        self.backbone = backbone
        self.layers_to_extract = layers_to_extract
        self.norm_mode = norm_mode #for testing how different normalization methods work
        self.hooked_features = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        #registers all the forward hooks in the layers_to_extract list
        for name, module in self.backbone.named_modules(remove_duplicate=False):
            if name in self.layers_to_extract:
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name):
        def hook(module, input_args, output):
            self.hooked_features[layer_name] = output
        return hook

    def _align_and_concatenate(self, feats):
        #upsample all the hooked feature maps to whatever the highest spatial resolution is
        #and then concatenate them on the channel dimension
        target = max((f.shape[-2:] for f in feats), key=lambda s: s[0] * s[1])
        feats = [f if f.shape[-2:] == target else
                 functional.interpolate(f, size=target, mode="bilinear", align_corners=False)
                 for f in feats]
        return torch.cat(feats, dim=1)

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.hooked_features = {}

    def forward(self, in_tensor: torch.Tensor):
        #pass the in_tensor to the backbone and process the hooked features
        self.hooked_features = {}

        with torch.no_grad():
            _ = self.backbone(in_tensor) #we just need to do a forward pass for the hooks to work
        feats = [self.hooked_features[n] for n in self.layers_to_extract]

        if self.norm_mode == "joint":
            maps = functional.normalize(self._align_and_concatenate(feats), dim=1)
            pooled = maps.mean(dim=(2, 3))
        elif self.norm_mode == "layer_pixel":
            maps = self._align_and_concatenate([functional.normalize(f, dim=1) for f in feats])
            pooled = maps.mean(dim=(2, 3))
        else:  # layer_pooled
            pooled = torch.cat([functional.normalize(f.mean(dim=(2, 3)), dim=1)
                                for f in feats], dim=1)
        return functional.normalize(pooled.float(), dim=1)