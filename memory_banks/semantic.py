from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import clip

class SemanticMemoryBank(nn.Module):
    #memory bank based on the semantic information extracted from CLIP
    def __init__(self,
                 category: str,
                 clip_model_name: str = "ViT-B/32"):
        super().__init__()

        self.model, _ = clip.load(clip_model_name,device="cpu")
        self.category = category
        self.register_buffer("memory_bank", torch.tensor([]))
        text_prompts = [
            f"a photo of a normal {category}",
            f"a photo of an anomalous {category}"
        ]
        text_inputs = clip.tokenize(text_prompts)
        self.register_buffer("text_tokens", text_inputs)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("normal_text_embeddings", text_features[0].unsqueeze(0))
        self.register_buffer("normal_image_embeddings", torch.empty(0, text_features.shape[1]))
        self.register_buffer("max_train_distance", torch.tensor(0.0))
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def build(self, images: torch.Tensor):
        images = self._preprocess_images(images)
        with torch.no_grad():
            #ensures images are on the same device as the model
            images = images.to(next(self.model.parameters()).device)
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        self.normal_image_embeddings = image_features

        with torch.no_grad():
            text_similarity = (100.0 * self.normal_image_embeddings @ self.normal_text_embeddings.T)
            text_probs = text_similarity.softmax(dim=-1)
            raw_scores = 1.0 - text_probs[:, 0]

        max_train_score = raw_scores.max().item()
        self.max_train_distance = torch.tensor(max(max_train_score, 1e-8))



    def score(self, test_image: torch.Tensor):
        with torch.no_grad():
            test_image = test_image.to(next(self.model.parameters()).device)
            test_features = self.model.encode_image(test_image)
            test_features = test_features / test_features.norm(dim=-1, keepdim=True)

        text_similarity = (100.0 * test_features @ self.normal_text_embeddings.T)
        text_probs = text_similarity.softmax(dim=-1)

        raw_score = 1.0 - text_probs[:, 0]
        normalized_score = raw_score / self.max_train_distance.item()
        normalized_score = torch.clamp(normalized_score, 0.0, 1.0)



        return raw_score, normalized_score

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images = functional.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
        images = images / 255.0
        images = (images - self.clip_mean) / self.clip_std

        return images
