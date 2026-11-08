from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as functional
def knn_score(query: torch.Tensor, bank: torch.Tensor, k: int = 1) -> torch.Tensor:
    #Mean distance to the k nearest bank entries. query [B,D], bank [N,D] -> [B]
    d = torch.cdist(query, bank)
    return d.topk(min(k, bank.shape[0]), largest=False).values.mean(dim=1)

class GlobalKNNDetector(nn.Module):
    #stores the global feature compositions of an image by averaging the pixel-level feature vectors

    def __init__(self,
                 feature_extractor: torch.nn.Module,
                 k: int=1
                 ):
        super().__init__()
        #feature channels refers to the number of channels in the fgeature maps from the backbone
        self.feature_extractor = feature_extractor.eval()
        self.register_buffer("memory_bank", torch.empty(0))
        self.k = k

    @property
    def device(self):
        #just makes it so that the device is a property of the object
        return next(self.feature_extractor.parameters()).device

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        #images should be shape [B, 3, H, W]. output tensor is [B,C] unit-norm global descriptors
        features = self.feature_extractor(images.to(self.device))
        return functional.normalize(features.mean(dim=(2,3)))

    @torch.no_grad()
    def embed_loader(self, loader):
        #Embed a whole loader once; cache the result for ablations.
        embeds = [self.embed(batch["image"]) for batch in loader]
        return torch.cat(embeds)

    def build(self,loader):
        #features should be of shape [N, C, H, W]
        self.memory_bank = self.embed_loader(loader)
        print(f"Composition memory bank built with {self.memory_bank.shape[0]} samples")

    @torch.no_grad()
    def score(self, images: torch.Tensor, k: int | None = None) -> torch.Tensor:
        #literally just do a knn check on the memory bank
        return knn_score(self.embed(images), self.memory_bank, k or self.k)