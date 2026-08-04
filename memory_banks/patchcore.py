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

class KCenterGreedy:
    #coreset subsampling algorithm for selection of normal patches
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float):
        self.embedding = embedding.detach().cpu().numpy()
        self.sampling_ratio = sampling_ratio
        self.num_samples = int(len(self.embedding) * sampling_ratio)
        self.min_distances = None

    def _pairwise_distances(self, x: np.ndarray, y: np.ndarray | None = None):
        if y is None:
            y = x
        x_norm = (x ** 2).sum(axis=1).reshape(-1, 1)
        y_norm = (y ** 2).sum(axis=1).reshape(1, -1)
        distances = x_norm + y_norm - 2.0 * np.dot(x,y.T)
        distances = np.maximum(distances, 0.0) #floating point cleanup
        return np.sqrt(distances)

    def sample_coreset(self):
        #iteratively select the point furthest away from already chosen points
        if self.num_samples >= len(self.embedding):
            return torch.tensor(self.embedding)

        selected_indices = [np.random.randint(0, len(self.embedding))]
        self.min_distances = self._pairwise_distances(self.embedding, self.embedding[selected_indices]).min(axis=1)

        for _ in range(1, self.num_samples):
            #iteratively select maximum minimum distance
            new_idx = np.argmax(self.min_distances)
            selected_indices.append(new_idx)

            #update minimum distances
            new_distances = self._pairwise_distances(self.embedding, self.embedding[selected_indices])
            self.min_distances = np.minimum(self.min_distances, new_distances.min(axis=1))

        selected_embeddings = self.embedding[selected_indices]
        return torch.tensor(selected_embeddings, dtype=torch.float32)

class PatchMemoryBank:
    def __init__(self,
                 num_neighbors = 9,
                 sampling_ratio = 0.1,
                 target_image_size = (256,256)):
        self.num_neighbors = num_neighbors
        self.sampling_ratio = sampling_ratio
        self.target_image_size = target_image_size
        self.memory_bank: torch.Tensor | None = None
        self.blur = GaussianBlur2d(kernel_size=33, sigma=4.0)

    def build(self, embeddings: torch.Tensor):
        #builds the memory bank from embeddings
        if self.sampling_ratio < 1.0:
            self.memory_bank = self._coreset_subsample(embeddings, self.sampling_ratio)
        else:
            self.memory_bank = embeddings

        print(f"Patch memory bank built with {self.memory_bank.shape[0]} patches")

    def _coreset_subsample(self, embeddings: torch.Tensor, sampling_ratio: float):
        sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=sampling_ratio)
        return sampler.sample_coreset()

    def score(self,
              test_embeddings: torch.Tensor,
              feature_map_shape: Tuple[int,int]
              )-> Tuple[torch.T, float]:
        #computes the path-level anomaly scores and then aggregates them
        device = test_embeddings.device
        memory_bank = self.memory_bank.to(device) #just make sure that all the tensors are on the same device

        distances = torch.cdist(test_embeddings, memory_bank, p=2.0)
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
        anomaly_map_smoothed = self.blur(anomaly_map_upscaled)

        anomaly_score = torch.max(min_distances).item() / 10.0 #calculate the image-level score by max pooling over spatial dimensions

        return anomaly_map_smoothed.squeeze().cpu(), anomaly_score