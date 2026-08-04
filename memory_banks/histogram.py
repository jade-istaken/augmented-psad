from typing import Tuple

import torch
import torch.nn.functional as functional

class HistogramMemoryBank:
    #class histogram memory bank (stores the distribution of classes in the normal images
    #used for comparison of class distribution is test images
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.memory_bank: torch.Tensor | None = None
        self.max_train_distance = 0.0 #used for the adaptive scaling of scores later on

    def build(self, masks: torch.Tensor):
        #build the memory bank from normal masks
        histograms = []

        for mask in masks:
            counts = torch.bincount(mask.flatten(), minlength=self.num_classes) #the minlength is so that all tensors are the same length even if the mask lacks certain classes
            proportions = counts / (counts.sum() + 1e-8) #normalizing the proportions of classes so that it adds up to 1.0
            histograms.append(proportions)

        self.memory_bank = torch.stack(histograms,dim=0) #stack all the histograms into a new tensor which acts as the memory bank
        self._compute_adaptive_scaling() #once the memory banks are populated pre-compute the adaptive scaling factor
        print(f"Histogram memory bank built with {self.memory_bank.shape[0]} histograms")

    def _compute_adaptive_scaling(self):
        #compute the maximum anomaly score within the normal samples
        #this is used to later normalize test scores to [0,1]
        if self.memory_bank is None or len(self.memory_bank) == 0:
            #in the case of an empty memory bank the max_train_distance simply becomes 1 as normalization doesn't occur
            self.max_train_distance = 1.0
            return

        max_dist = 0.0
        for i in range(len(self.memory_bank)):
            test_hist = self.memory_bank[i].unsqueeze(0)
            distances = torch.cdist(test_hist, self.memory_bank, p=2.0) #get the euclidean distances between the candidate histogram and all other histograms
            min_dist = distances.min().item()
            if min_dist > max_dist:
                max_dist = min_dist

        self.max_train_distance = min(max_dist, 1e-8) #add a lower bound so that there isn't divide by zero when it comes to scaling

    def score(self, test_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        counts = torch.bincount(test_mask.flatten(), minlength=self.num_classes).float()
        test_proportions = counts / (counts.sum() + 1e-8) #compute and normalize test histogram

        device = test_proportions.device
        memory_bank = self.memory_bank.to(device) #move memory bank to the same device as test tensor

        test_proportions = test_proportions.unsqueeze(0) #expected shape is [1, num_classes]
        distances = torch.cdist(test_proportions, memory_bank, p=2.0) #expected shape is [1, N] where N is number of histograms

        raw_distance = distances.min().item()
        normalized_score = raw_distance / self.max_train_distance
        normalized_score = min(normalized_score, 1.0) #clamp to [0,1]

        return torch.tensor(raw_distance), normalized_score