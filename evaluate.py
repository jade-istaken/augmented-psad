import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import MVTecLOCODataLoader
from memory_banks import HistogramMemoryBank, PatchMemoryBank, CompositionMemoryBank
from models import init_backbone, FeatureExtractor, Segmenter

def visualize_anomaly_map(image_tensor:torch.Tensor, anomaly_map: torch.Tensor, save_path):
    #helper function to visualize anomaly masks
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.clip(img_np, 0, 1)
    print(f"Image stats - min: {img_np.min():.3f}, max: {img_np.max():.3f}, dtype: {img_np.dtype}")


    anomaly_map_np = anomaly_map.cpu().numpy()
    anomaly_map_np = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)

    fig, axes = plt.subplots(1,3,figsize=(15,5))

    axes[0].imshow(img_np, vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im = axes[1].imshow(anomaly_map_np, cmap='jet')
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_np)
    axes[2].imshow(anomaly_map_np, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size=1,
        num_workers=args.num_workers
    )

    test_loaders = data_manager.get_all_test_loaders()

    # initialize feature extraction model and segmentation model from the saved checkpoints
    backbone = init_backbone()
    feature_extractor = FeatureExtractor(backbone=backbone, layers_to_extract=["layer2", "layer3"]).to(device)
    feature_extractor.eval()

    seg_model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=False).to(device)
    seg_model.load_state_dict(torch.load(Path(args.seg_checkpoint)))
    seg_model.eval()

    # initialize and load saved memory banks
    bank_save_dir = Path(args.save_dir) / args.category
    memory_bank_states = torch.load(bank_save_dir / 'memory_banks.pth')

    hist_state = memory_bank_states['hist_bank']
    hist_bank = HistogramMemoryBank(num_classes=args.num_classes)
    hist_bank.memory_bank = hist_state.pop('memory_bank')
    hist_bank.load_state_dict(hist_state, strict=False)
    hist_bank.to(device)

    comp_state = memory_bank_states['comp_bank']
    comp_bank = CompositionMemoryBank(
        num_classes=args.num_classes,
        feature_channels=1792,
        feature_extractor=feature_extractor
    )
    comp_bank.memory_bank = comp_state.pop('memory_bank')
    comp_bank.load_state_dict(comp_state, strict=False)
    comp_bank.to(device)

    patch_state = memory_bank_states['patch_bank']
    patch_bank = PatchMemoryBank(
        num_neighbors=args.num_neighbors,
        sampling_ratio=args.sampling_ratio,
        target_image_size=(512, 512)
    )
    patch_bank.memory_bank = patch_state.pop('memory_bank')
    patch_bank.mean = patch_state.pop('mean')
    patch_bank.std = patch_state.pop('std')
    patch_bank.load_state_dict(patch_state, strict=False)
    patch_bank.to(device)

    viz_dir = Path(args.save_dir) / args.category
    viz_dir.mkdir(parents=True, exist_ok=True)
    max_viz_samples = 5

    print("Beginning test runs")
    with torch.no_grad():
        for atype in test_loaders:  # iterate through the test loaders one by one
            loader = test_loaders[atype]
            hist_anomaly_score = 0.0
            norm_hist_anomaly_score = 0.0
            comp_anomaly_score = 0.0
            norm_comp_anomaly_score = 0.0
            patch_anomaly_score = 0.0
            viz_count = 0
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Testing {atype}")):
                imgs = batch['image'].to(device)
                coords = batch['coord'].to(device)

                seg_logits = seg_model(imgs, coords)
                seg_masks = torch.argmax(seg_logits, dim=1)
                features = feature_extractor(imgs)
                B, C, H, W = features.shape

                hist_anomaly_scores = hist_bank.score(seg_masks)
                comp_anomaly_scores = comp_bank.score(features, seg_masks)
                flat_embeddings = features.permute(0, 2, 3, 1).reshape(-1, C)
                patch_anomaly_scores = patch_bank.score(flat_embeddings, (H, W))

                hist_anomaly_score+= hist_anomaly_scores[0]
                norm_hist_anomaly_score+=hist_anomaly_scores[1]
                comp_anomaly_score+= comp_anomaly_scores[0]
                norm_comp_anomaly_score+=comp_anomaly_scores[1]
                patch_anomaly_score+=patch_anomaly_scores[1]

                if viz_count < max_viz_samples:
                    #only visualize up to a max viz samples variable so that we're not flooding disk space for no reason
                    orig_img = imgs[0].cpu()

                    save_path = viz_dir / f"{atype}_sample_{batch_idx}.png"
                    visualize_anomaly_map(orig_img, patch_anomaly_scores[0], save_path)
                    viz_count += 1

            avg_hist_score = hist_anomaly_score / len(loader)
            avg_comp_score = comp_anomaly_score / len(loader)
            avg_patch_score = patch_anomaly_score / len(loader)
            norm_avg_hist_score = norm_hist_anomaly_score / len(loader)
            norm_avg_comp_score = norm_comp_anomaly_score / len(loader)

            print(f"Average {atype} raw anomaly scores: hist={avg_hist_score:.4f} | comp={avg_comp_score:.4f} | patch={avg_patch_score:.4f}")
            print(f"Average {atype} normalized anomaly scores: hist={norm_avg_hist_score:.4f} | comp={norm_avg_comp_score:.4f} | patch={avg_patch_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate model performance on test dataset")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--seg_checkpoint", type=str, required=True, help="Path to trained Phase 2 segmentation model")
    parser.add_argument("--save_dir", type=str, default="./processed_memory_banks",
                        help="Path to processed memory banks")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=9,
                        help="Number of neighbors for patchcore, should match number used during memory bank construction")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)

    args = parser.parse_args()
    evaluate(args)

    # category class numbers in mvtec_loco:
    # screw_bag = 7
    # breakfast_box = 7
    # juice_bottle = 9
    # pushpins = 26
    # splicing_connectors = 10