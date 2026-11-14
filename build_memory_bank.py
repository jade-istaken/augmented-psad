import torch
import argparse
import json
from pathlib import Path
from models.feature_extractor import FeatureExtractor
from models.backbone import init_backbone, get_norm_stats
from dataset import MVTecLOCODataLoader
from detector import GlobalKNNDetector

def build_bank(args, feature_extractor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    mean, std = get_norm_stats(args.backbone)
    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    if args.few_shot:
        train_loader = data_manager.get_few_shot_loader()
    else:
        train_loader = data_manager.get_train_loader()

    #frozen feature extractor backbone for memory bank
    if feature_extractor is None:
        backbone = init_backbone(args.backbone,args.layers)
        feature_extractor = FeatureExtractor(backbone=backbone, layers_to_extract=args.layers, norm_mode=args.norm_mode).to(device)
    feature_extractor.eval()

    #init the memory bank
    detector = GlobalKNNDetector(feature_extractor=feature_extractor)

    print("Building memory bank")
    detector.build(train_loader)


    save_dir = Path(args.save_dir) / args.category
    save_dir.mkdir(parents=True,exist_ok=True)
    torch.save(detector.memory_bank.cpu(), save_dir / "memory_bank.pt")
    config = {
        "category": args.category,
        "layers": args.layers,
        "embedding_dim": detector.memory_bank.shape[1],
        "num_images": detector.memory_bank.shape[0],
        "seed": args.seed,
        "torch": torch.__version__,
        "shot": "few_shot" if args.few_shot else "full_shot",
        "backbone": args.backbone,
        "norm_mode": args.norm_mode
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved bank {tuple(detector.memory_bank.shape)} to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Memory Banks")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True, help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./processed_memory_banks")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot loader")
    parser.add_argument("--backbone", type=str, default="CLIP")
    parser.add_argument("--norm_mode", type=str, default="layer_pooled")
    args = parser.parse_args()
    build_bank(args)