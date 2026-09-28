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
                 clip_model_name: str = "ViT-B/32",
                 context_length: int = 16):
        super().__init__()

        self.model, _ = clip.load(clip_model_name,device="cpu")
        self.model.eval()
        for param in self.model.parameters():
            #freeze the model because it's only being used for inference
            param.requires_grad = False
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

        self.context_length = context_length
        self.context_dimension = self.model.ln_final.weight.shape[0] # thii is the CLIP text embedding dimension
        self.learnable_context = nn.Parameter(torch.empty(self.context_length, self.context_dimension))
        nn.init.normal(self.learnable_context, std=0.02)



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
            test_image = self._preprocess_images(test_image)
            test_image = test_image.to(next(self.model.parameters()).device)
            test_features = self.model.encode_image(test_image)
            test_features = test_features / test_features.norm(dim=-1, keepdim=True)

        text_features = self._get_text_features()
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

    def _prepare_class_tokens(self):
        #extract and register fixed token embeddings for start of string, class name, and end of string tokens
        tokenized_category = clip.tokenize([self.category])
        eos_mask = tokenized_category == 49407
        eos_index = eos_mask.nonzero(as_tuple=True)[1][0].item()

        prefix_ids = tokenized_category[0, :1] #just the SOS token
        suffix_ids = tokenized_category[0, 1:eos_index+1] #suffix is class tokens and eos token

        with torch.no_grad():
            prefix_emb = self.model.token_embedding(prefix_ids.unsqueeze(0).to('cpu')).squeeze(0)
            suffix_emb = self.model.token_embedding(suffix_ids.unsqueeze(0).to('cpu')).squeeze(0)

        self.register_buffer("prefix_embeddings", prefix_emb)
        self.register_buffer("suffix_embeddings", suffix_emb)

    def _get_text_features(self) -> torch.Tensor:
        #injects learnable context directly into the transformer sequence
        device = next(self.model.parameters()).device()

        context = self.learnable_context.unsqueeze(0).to(device)
        prefix = self.prefix_embeddings.unsqueeze(0).to(device)
        suffix = self.suffix_embedding.unsqueeze(0).to(device)

        prompts =torch.cat([prefix, context, suffix], dim=1)

        seq_length = prompts.shape[1]
        if seq_length < 77:
            #pad to max length
            padding = torch.zeros(1, 77 - seq_length, prompts.shape[2], device=device)
            prompts = torch.cat([prompts, padding], dim=1)

        prompts = self.model.positional_embedding.unsqueeze(0).to(device)
        prompts = prompts.permute(1,0,2)

        x = self.model.transformer(prompts)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)

        eos_pos = prefix.shape[1] + self.context_length + suffix.shape[1] -1
        x = x[torch.arange(x.shape[0]), eos_pos] @ self.model.text_projection.to(device)

        return x / x.norm(dim=-1, keepdim=True)

    def train_prompts(self, images: torch.Tensor, epochs: int = 50, lr: float = 0.001):
        images = self._preprocess_images(images)
        device = next(self.model.parameters()).device
        images = images.to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        optimizer = torch.optim.AdamW([self.learnable_context], lr=lr)

        #all the training images are "normal" so we want the target tensor to be all 0
        target = torch.zeros(images.shape[0], dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()

        self.train()

        for epoch in range(epochs):
            #training loop to refine the learnable prompts
            optimizer.zero_grad()

            text_features = self._get_text_features() #get current state of learnable prompt
            text_features = text_features.expand(images.shape[0], -1)
            logits = (100.0 * image_features @ text_features.T)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        self.eval()