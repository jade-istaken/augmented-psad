from typing import Dict

import torch

class AdaptiveScaler:
    #normalize the anomaly scores based on the maximum score observed in normal data

    def __init__(self):
        self.max_scores: Dict[str, float] = {}

    def fit(self, bank_name: str, raw_scores: torch.Tensor):
        max_val = raw_scores.max().item()
        self.max_scores[bank_name] = max(max_val, 1e-8) #prevent division by zero
        print(f"Scaler fitted for {bank_name}: max_score = {self.max_scores[bank_name]:.4f}")

    def normalize(self, bank_name: str, raw_score: float) -> float:
        max_val = self.max_scores.get(bank_name, 1.0)
        normalized = raw_score / max_val
        return min(normalized, 1.0) #clamp to [0,1]

    def combine_scores(self, scores: Dict[str, float]) -> float:
        return sum(scores.values())