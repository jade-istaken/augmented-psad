#THE BIG ONE
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class CompositionMemoryBank(nn.Module):
    #stores the feature compositions of classes by averaging the pixel-level feature vectors of each class

    def __init__(self,
                 num_classes: int,
                 feature_channels: int,
                 feature_extractor: torch.nn.Module
                 ):
        super().__init__()
        #feature channels refers to the number of channels in the fgeature maps from the backbone
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.feature_extractor = feature_extractor
        #self.memory_bank: torch.Tensor | None = None
        self.register_buffer("memory_bank", torch.tensor([]))
        self.register_buffer("max_train_distance", torch.tensor(0.0))
        self.present_classes: set = set() #set used to track which classes are present in the training data

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        #images should be shape [B, 3, H, W]. output tensor is [B, C, H, W]
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)

        return features

    def compute_class_embeddings(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        #compute the class embeddings by averaging the per-class features
        #features should be shape [B, C, H, W]. Masks should be shape [B,H,W]
        #the output composition vectors are going to be [B, num_classes * C]
        batch_size, channels, height, width = features.shape
        device = features.device

        #resize mask resolution if needed
        if masks.shape[-2:] != (height, width):
            masks_resized = functional.interpolate(
                masks.unsqueeze(1).float(),
                size=(height, width),
                mode="nearest"
            ).squeeze(1).long()
        else:
            masks_resized = masks

        composition_vectors = []

        for b in range(batch_size):
            class_embeddings = []
            feature_map = features[b] #[C,H,W]
            mask = masks_resized[b] #[H,W]

            #where we're going, we don't need spatial resolution
            feature_flat = feature_map.view(channels, -1).t() #[H*W, C]
            mask_flat= mask.view(-1) #[H*W]

            for cls in range(self.num_classes):
                class_mask = mask_flat == cls #find all pixels belonging to a class
                if class_mask.sum() > 0:
                    class_features = feature_flat[class_mask]
                    class_embedding = class_features.mean(dim=0)
                    self.present_classes.add(cls)
                else:
                    #if the class isn't present just use a zero vector
                    class_embedding = torch.zeros(channels, device=device)

                class_embeddings.append(class_embedding)
            #concatenate the class embeddings and append to the list
            composition_vector = torch.cat(class_embeddings,dim=0) #[num_classes * C]
            composition_vectors.append(composition_vector)
        return torch.stack(composition_vectors, dim=0) #stack and return the list of vectors, shape is [B, num_classes*C

    def build(self,
              features: torch.Tensor,
              masks: torch.Tensor
              ):
        #features should be of shape [N, C, H, W] and maks should; be [B,H,W]


        composition_vectors = self.compute_class_embeddings(features,masks)
        print(f"Composition vectors shape: {composition_vectors.shape}")

        self.memory_bank = composition_vectors
        self.memory_bank = functional.normalize(self.memory_bank, p=2, dim=1) #normalize the composition vectors for easy comparisons later
        self._compute_adaptive_scaling()

        print(f"Composition memory bank built with {self.memory_bank.shape[0]} samples")
        print(f"Classes present in the training are {sorted(self.present_classes)}")

    def _compute_adaptive_scaling(self):
        # compute the maximum anomaly score within the normal samples
        # this is used to later normalize test scores to [0,1]
        if self.memory_bank is None or len(self.memory_bank) < 2:
            self.max_train_distance = torch.tensor(1.0)
            return

        max_dist = 0.0

        for i in range(len(self.memory_bank)):
            test_vec = self.memory_bank[i].unsqueeze(0)
            distances = torch.cdist(test_vec, self.memory_bank, p=2.0)
            min_dist = distances.min().item()
            if min_dist > max_dist:
                max_dist = min_dist

        self.max_train_distance = torch.tensor(max(max_dist, 1e-8))

    def score(self,
              features: torch.Tensor,
              mask: torch.Tensor
              ) -> Tuple[torch.Tensor,float]:
        #features should be [1, C, H, W] and mask should be [1,H,W]
        #ensure correct dimensions first
        if features.dim() == 3:
            features = features.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        device = features.device
        memory_bank = self.memory_bank.to(device)  # move memory bank to same device as image tensor

        composition_vector = self.compute_class_embeddings(features, mask) #compute the composition vector
        composition_vector = functional.normalize(composition_vector, p=2, dim=1) #normalize it

        distances = torch.cdist(composition_vector, self.memory_bank,p=2.0)
        raw_distance = distances.min().item()

        normalized_score = raw_distance / self.max_train_distance.item()
        normalized_score = min(normalized_score, 1.0) #clamp to [0,1]

        return torch.Tensor(raw_distance), normalized_score