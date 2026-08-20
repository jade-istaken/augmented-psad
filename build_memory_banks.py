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
    seg_model_path = Path(args.seg_checkpoint)
    seg_model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(Path(args.seg_checkpoint)))
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
        target_image_size=(512,512),
        fast_dev_mode=args.random_patches
    )
    scaler = AdaptiveScaler()

    #begin collecting the relevant data for the banks
    print("Extracting features and masks from normal training data")
    all_masks = []
    all_composition_features = [] #List[(features)]
    all_patch_embeddings = []

    with torch.no_grad():
        for batch in tqdm(train_loader):
            imgs = batch['image'].to(device)
            coords = batch['coord'].to(device)

            seg_logits = seg_model(imgs, coords)
            seg_masks = torch.argmax(seg_logits, dim=1)  # [B,H,W]

            features = feature_extractor(imgs)  # [B,C,H,W]

            seg_masks_cpu = seg_masks.cpu() #moved to cpu immediately to free vram
            features_cpu = features.cpu()

            all_masks.append(seg_masks_cpu)
            all_composition_features.append(features_cpu)

            #flatten features for patch bank
            B, C, H, W = features_cpu.shape
            flat_embeddings = features_cpu.permute(0, 2, 3, 1).reshape(-1, C)
            all_patch_embeddings.append(flat_embeddings)
            del imgs, coords, seg_logits, seg_masks, features #explicitly delete these to stop over-caching

    #build the banks
    hist_masks = torch.cat(all_masks, dim=0)

    #just reuse the mask tensor for space saving
    comp_masks = hist_masks
    del all_masks  # free the list immediately
    # concatenate the features
    comp_features = torch.cat(all_composition_features, dim=0)
    del all_composition_features
    #concatenate patches
    patch_embeddings = torch.cat(all_patch_embeddings, dim=0)
    del all_patch_embeddings

    #lkets just force garbage collection too
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # build the banks
    print("Building histogram memory bank")
    hist_bank.build(hist_masks)

    print("Building composition memory bank")
    comp_bank.build(comp_features, comp_masks)

    print("Building patch memory bank (This will take a while due to coreset subsampling)")
    patch_bank.build(patch_embeddings)

    #Record the adaptive sfcaling statistics into the scaler
    print("Storing the adaptive scaling statistics")

    scaler.max_scores['hist'] = hist_bank.max_train_distance
    scaler.max_scores['comp'] = comp_bank.max_train_distance
    scaler.max_scores['patch'] = patch_bank.max_train_distance

    save_dir = Path(args.save_dir) / args.category
    save_dir.mkdir(parents=True,exist_ok=True)

    save_dict = {
        'patch_bank': patch_bank.state_dict(),
        'hist_bank': hist_bank.state_dict(),
        'comp_bank': comp_bank.state_dict(),
        'scaler': scaler
    }

    torch.save(save_dict, save_dir / 'memory_banks.pth')
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
    parser.add_argument("--random_patches", type=bool, default=False, help="Whether to use random sampling to speed up patchcore building")

    args = parser.parse_args()
    build_banks(args)

    #category class numbers in mvtec_loco:
    #screw_bag = 7
    #breakfast_box = 7
    #juice_bottle = 9
    #pushpins = 26
    #splicing_connectors = 10