import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class GaussianBlur2d(nn.Module):
    kernel: torch.Tensor #explicitly set buffer type to appease statis analysis

    def __init__(self, kernel_size = 21, sigma = 4.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        range_tensor = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(range_tensor ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        self.register_buffer('kernel', kernel_1d.outer(kernel_1d).unsqueeze(0).unsqueeze(0))

    def forward(self, input_tensor: torch.Tensor):
        padding = self.kernel_size // 2
        return functional.conv2d(input_tensor, self.kernel, padding=padding)

class SparseRandomProjection(nn.Module):
    #reimplenting johnson-lindenstrauss SRP like from scikit but torch native
    def __init__(self, epsilon: float = 0.9):
        super().__init__()
        self.epsilon = epsilon
        self.num_components: int = 0
        self.register_buffer("components_", torch.tensor([]))
        self.is_fitted: bool = False

    def fit(self, input_tensor: torch.Tensor):
        num_samples, num_features = input_tensor.shape

        #set number of components to the minimum bound of the j-l lemma (num_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3))
        denominator = (self.epsilon ** 2 / 2.0) - (self.epsilon ** 3 / 3.0)
        numerator = 4 * math.log(num_samples)
        num_components = int(math.ceil(numerator/denominator)) #use a ceiling operation to make sure that it never undershoots component count

        self.num_components = min(num_components, num_features) #if the number of features is less than the theoretical bound we can just use the number of features

        #initialize the projection matrix with a normal distribution that's been scaled by 1/sqrt(num_components) to preserve distance
        new_components = torch.randn(
            num_features, self.num_components, dtype=input_tensor.dtype, device=input_tensor.device
        ) / math.sqrt(self.num_components)

        self.register_buffer("components_", new_components)
        self.is_fitted = True

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise ValueError("please fit the model before transform by calling fit() first")
        return torch.matmul(input_tensor, self.components_)

class KCenterGreedy:
    #coreset subsampling algorithm for selection of normal patches
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float):
        self.embedding = embedding.detach()
        self.sampling_ratio = sampling_ratio
        self.num_samples = int(len(self.embedding) * sampling_ratio)
        self.projection = SparseRandomProjection(epsilon=0.9)

        self.features: torch.Tensor | None = None
        self.features_normal_squared: torch.Tensor | None = None
        self.min_distances: torch.Tensor | None = None

    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor | None = None):
        if y is None:
            y = x
        return torch.cdist(x,y, p=2.0)

    def sample_coreset(self) -> torch.Tensor:
        #iteratively select the point furthest away from already chosen points
        if self.num_samples >= len(self.embedding):
            return self.embedding.clone()

        #fit the embeddings to a lower dimensional space so it stops crashing immediately upon call
        print("beginning projection")
        self.projection.fit(self.embedding)
        print("projection fitted")
        self.features = self.projection.transform(self.embedding).detach()
        self.features_normal_squared = torch.sum(self.features ** 2, dim=1)
        print("projection transformed")

        first_idx = torch.randint(0, len(self.features), (1,)).item() #changed to use torch.randint to improve reproduceability
        selected_indices = [first_idx]

        first_center = self.features[first_idx].squeeze()
        first_center_normal_squared = torch.sum(first_center ** 2)
        print("calculated first center")
        dot_product = torch.matmul(self.features, first_center)
        dist_squared = self.features_normal_squared + first_center_normal_squared - 2 * dot_product
        self.min_distances = torch.sqrt(torch.clamp(dist_squared, min=0.0)).detach()
        print("calculated first distance pair")

        for i in range(1, self.num_samples):

            new_idx = torch.argmax(self.min_distances).item()
            selected_indices.append(new_idx)


            new_center = self.features[new_idx]
            new_center_normal_squared = torch.sum(new_center ** 2)
            dot_product = torch.matmul(self.features, new_center)
            dist_squared = self.features_normal_squared + new_center_normal_squared - 2 * dot_product

            new_distances = torch.sqrt(torch.clamp(dist_squared, min=0.0)).detach()

            self.min_distances = torch.minimum(self.min_distances, new_distances).detach()
            print(f"pair {i} of {self.num_samples}, distance: {new_distances}")
            if i % 10 == 0:
                if self.features.is_cuda:
                    torch.cuda.empty_cache()

        return self.embedding[selected_indices]

class PatchMemoryBank(nn.Module):
    def __init__(self,
                 num_neighbors = 9,
                 sampling_ratio = 0.1,
                 target_image_size = (256,256),
                 fast_dev_mode: bool = False,
                 batch_size: int = 10000):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.sampling_ratio = sampling_ratio
        self.target_image_size = target_image_size
        self.blur = GaussianBlur2d(kernel_size=33, sigma=4.0)
        self.batch_size = batch_size
        self.max_train_distance = 1.0
        self.fast_dev_mode = fast_dev_mode

        # self.memory_bank: torch.Tensor | None = None
        # self.mean: torch.Tensor | None = None
        # self.std: torch.Tensor | None = None
        #register all these as buffers so they move to devices easily and also save in the state_dict
        self.register_buffer("memory_bank", torch.tensor([]))
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("std", torch.tensor([]))

    def build(self, embeddings: torch.Tensor):
        #builds the memory bank from embeddings
        if self.sampling_ratio < 1.0:
            if self.fast_dev_mode:
                print("Fast dev mode enabled: Using pure random subsampling.")
                self.memory_bank = self._random_subsample(embeddings, self.sampling_ratio)
            else:
                print("Beginning coreset subsampling")
                self.memory_bank = self._coreset_subsample(embeddings, self.sampling_ratio)
        else:
            self.memory_bank = embeddings.clone()

        self._standardize_memory_bank()
        print(f"Patch memory bank built with {self.memory_bank.shape[0]} patches")

    def _coreset_subsample(self, embeddings: torch.Tensor, sampling_ratio: float):
        sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=sampling_ratio)
        return sampler.sample_coreset()

    def _random_subsample(self, embeddings: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
        #random subsampling for the purposes of faster dev because doing real subsampling takes like 3 hours
        num_samples = int(len(embeddings) * sampling_ratio)
        indices = torch.randperm(len(embeddings))[:num_samples]
        return embeddings[indices]

    def _compute_adaptive_scaling(self, embeddings: torch.Tensor) -> float:
        #compute the max nearest neighbor distance across all training batches
        device = embeddings.device
        max_dist = 0.0
        memory_bank = self.memory_bank.to(device)

        #iterate through embeddings in batches to avoid out of memory errors
        for i in range(0, embeddings.shape[0], self.batch_size):
            batch = embeddings[i:i+self.batch_size].to(device)
            distances = torch.cdist(batch, memory_bank, p=2.0) #shape: [batch_size, memory_bank_size]
            min_dists = distances.min(dim=1)[0]

            batch_max = min_dists.max().item()
            if batch_max > max_dist:
                max_dist = batch_max
        return max(max_dist, 1e-8)

    def _standardize_memory_bank(self):
        self.mean = self.memory_bank.mean(dim=0, keepdim=False)
        self.std = self.memory_bank.std(dim=0, keepdim=False) + 1e-8 # (add a tiny epsilon factor for later division)
        self.memory_bank = (self.memory_bank - self.mean) / self.std

    def score(self,
              test_embeddings: torch.Tensor,
              feature_map_shape: Tuple[int,int]
              )-> Tuple[torch.Tensor, float]:
        #computes the path-level anomaly scores and then aggregates them
        device = test_embeddings.device

        test_embeddings_norm = (test_embeddings - self.mean.to(device)) / self.std.to(device) #normalized embeddings
        memory_bank = self.memory_bank.to(device) #just make sure that all the tensors are on the same device

        distances = torch.cdist(test_embeddings_norm, memory_bank, p=2.0)
        patch_scores, _ = distances.topk(k=self.num_neighbors, largest=False,dim=1) #distance to k nearest neighbors
        min_distances = patch_scores[:,0] #the distance to the nearest neighbor is the primary patch score

        height,width = feature_map_shape #reshape back to 2d
        anomaly_map_2d = min_distances.reshape(1,1,height,width)

        anomaly_map_upscaled = functional.interpolate(
            anomaly_map_2d,
            size=self.target_image_size,
            mode="bilinear",
            align_corners=False
        )
        self.blur = self.blur.to(device)
        anomaly_map_smoothed = self.blur(anomaly_map_upscaled)

        anomaly_score = torch.max(min_distances).item()

        return anomaly_map_smoothed.squeeze().cpu(), anomaly_score