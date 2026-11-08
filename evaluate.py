import json

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset import MVTecLOCODataLoader
from detector import GlobalKNNDetector
from models import init_backbone, FeatureExtractor
from sklearn.metrics import roc_auc_score
from utils.metrics import compute_metrics

def image_auroc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    return float(roc_auc_score(y, np.r_[normal, anomalous]))


def score_loader(detector, loader, desc) -> np.ndarray:
    scores = [detector.score(b["image"]).cpu() for b in tqdm(loader, desc=desc)]
    return torch.cat(scores).numpy()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank_save_dir = Path(args.save_dir) / args.category
    cfg = json.loads((bank_save_dir / "config.json").read_text())

    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    loaders = data_manager.get_all_test_loaders() # returns a dictionary with keys ['good', 'logical_anomalies', 'structural_anomalies']

    # initialize feature extraction model and segmentation model from the saved checkpoints
    backbone = init_backbone()
    feature_extractor = FeatureExtractor(backbone=backbone, layers_to_extract=cfg["layers"]).to(device)
    feature_extractor.eval()

    # initialize and load saved memory banks
    detector = GlobalKNNDetector(
        feature_extractor=feature_extractor,
        k=args.num_neighbors
    )
    detector.memory_bank = torch.load(bank_save_dir / "memory_bank.pt").to(device)


    scores = {t: score_loader(detector, l, t) for t, l in loaders.items()}

    results = {
        "category": args.category,
        "k": args.num_neighbors,
        "layers": cfg["layers"],
        "LA_auroc": image_auroc(scores["good"], scores["logical_anomalies"]),
        "SA_auroc": image_auroc(scores["good"], scores["structural_anomalies"]),
        "mean_score": {t: float(s.mean()) for t, s in scores.items()},
        "num_images": {t: int(len(s)) for t, s in scores.items()},
    }

    print("Beginning test runs")
    out = bank_save_dir / f"results_k{args.num_neighbors}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate model performance on test dataset")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")

    parser.add_argument("--save_dir", type=str, default="./processed_memory_banks",
                        help="Path to processed memory banks")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=1, help="Number of neighbors")
    args = parser.parse_args()
    evaluate(args)
