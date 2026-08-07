import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from models.segmentation import Segmenter
from models.feature_extractor import FeatureExtractor
from models.backbone import init_backbone
from data.dataset import MVTecLOCODataLoader
from memory_banks.histogram import HistogramMemoryBank
from memory_banks.patchcore import PatchMemoryBank
from memory_banks.composition import CompositionMemoryBank
from utils.scaling import AdaptiveScaler

def build_banks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader = data_manager.get_train_loader()

    #initialize models
    #trained segmentation model from phase 2 (refine_unet.py)
    seg_model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=False).to(device)
    seg_model.eval()

    #frozen feature extractor backbone for memory banks
    backbone = init_backbone()
    feature_extractor = FeatureExtractor(backbone=backbone, layers_to_extract=["layer2","layer3"]).to(device)
    feature_extractor.eval()

    #init the memory bank s
    hist_bank = HistogramMemoryBank(num_classes=args.num_classes)
    comp_bank = CompositionMemoryBank(
        num_classes=args.num_classes,
        feature_channels = 1792, #for wideresnet-101 layer2 has 768 and layer3 has 1024
        feature_extractor=feature_extractor
    )
    patch_bank = PatchMemoryBank(
        num_neighbors=args.num_neighbors,
        sampling_ratio=args.sampling_ratio,
        target_image_size=(512,512)
    )
    scaler = AdaptiveScaler()

    #begin collecting the relevant data for the banks
    print("Extracting features and masks from normal training data")
    all_masks = []
    all_composition_data = [] #List[(features,masks)]
    all_patch_embeddings = []

    with torch.no_grad():
        for batch in tqdm(train_loader):
            imgs = batch['image'].to(device)
            coords = batch['coord'].to(device)

            seg_logits = seg_model(imgs, coords)
            seg_masks = torch.argmax(seg_logits, dim=1) #[B,H,W]
            all_masks.append(seg_masks.cpu())

            features = feature_extractor(imgs) # [B,C,H,W]
            all_composition_data.append((features.cpu(), seg_masks.cpu()))

            #flatten features for patch bank
            B,C,H,W = features.shape
            flat_embeddings = features.permute(0,2,3,1).reshape(-1,C)
            all_patch_embeddings.append(flat_embeddings.cpu())

    #build the banks
    print("Building histogram memory bank")
    hist_masks = torch.cat(all_masks, dim=0)
    hist_bank.build(hist_masks)

    print("building composition memory bank")
    #concatenate the features and masks
    comp_features = torch.cat([data[0] for data in all_composition_data], dim=0)
    comp_masks = torch.cat([data[1] for data in all_composition_data], dim=0 )
    comp_bank.build(comp_features, comp_masks)

    print("Building patch memory bank (Probably going to take a while because of coreset subsampling)")
    patch_embeddings = torch.cat(all_patch_embeddings)
    patch_bank.build(patch_embeddings)

    #Record the adaptive sfcaling statistics into the scaler
    print("Storing the adaptive scaling statistics")

    scaler.max_scores['hist'] = hist_bank.max_train_distance
    scaler.max_scores['comp'] = comp_bank.max_train_distance
    scaler.max_scores['patch'] = patch_bank.max_train_distances

    save_dir = Path(args.save_dir) / args.category
    save_dir.mkdir(parents=True,exist_ok=True)

    torch.save({
        'hist_bank': hist_bank,
        'comp_bank': comp_bank,
        'patch_bank': patch_bank,
        'scaler': scaler
    }, save_dir / 'memory_banks.pth')
    print(f"Memory banks and scaler saved to {save_dir / 'memory_banks.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Memory Banks")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True, help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--seg_checkpoint", type=str, required=True, help="Path to trained Phase 2 segmentation model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=9)
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./processed_memory_banks")

    args = parser.parse_args()
    build_banks(args)

    #category class numbers in mvtec_loco:
    #screw_bag = 7
    #breakfast_box = 7
    #juice_bottle = 9
    #pushpins = 26
    #splicing_connectors = 10