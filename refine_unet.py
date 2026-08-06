import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import numpy as np
import os
from models.segmentation import Segmenter
from data.dataset import MVTecLOCODataLoader, MVTecLOCODataset

COLOR_PALETTE = np.array([[0, 0, 0],
[204, 241, 227],
[112, 142, 18],
[254, 8, 23],
[207, 149, 84],
[202, 24, 214],
[230, 192, 37],
[241, 80, 68],
[74, 127, 0],
[2, 81, 216],
[24, 240, 129],
[20, 215, 125],
[161, 31, 204],
[254, 52, 116],
[117, 198, 203],
[4, 41, 68],
[127, 252, 61],
[21, 3, 142],
[40, 10, 159],
[241, 61, 36],
[14, 175, 77],
[144, 61, 115],
[131, 79, 97],
[109, 177, 163],
[58, 198, 140],
[17, 235, 168],
[47, 128, 91],
[238, 103, 45],
[124, 35, 228],
[101, 48, 232],
[74, 124, 114],
[78, 49, 30],
[35, 167, 27],
[137, 231, 47],
[235, 32, 39],
[56, 112, 32],
[62, 173, 79],
[86, 44, 201],
[77, 47, 217],
[246, 223, 57]])

def save_pseudo_label_samples(images: torch.Tensor, logits: torch.Tensor, epoch:int, save_dir:str="./pseudo_label_samples"):
    os.makedirs(save_dir, exist_ok=True)

    images_np = images.detach().cpu().clamp(0,1).numpy()
    images_np = (images_np *255).astype(np.uint8).transpose(0,2,3,1)

    preds = torch.argmax(logits,dim=1).detach().cpu().numpy()

    for i in range(images_np.shape[0]):
        img_pil = Image.fromarray(images_np[i]) #convert to PIL image

        mask_rgb = COLOR_PALETTE[preds[i]] #map class indices to rgb colors
        mask_pil = Image.fromarray(mask_rgb)

        total_width = img_pil.width + mask_pil.width
        max_height = max(img_pil.height, mask_pil.height)
        combined = Image.new('RGB', (total_width, max_height))
        combined.paste(img_pil, (0, 0))
        combined.paste(mask_pil, (img_pil.width, 0)) #concatenate mask and original horizontally

        path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
        combined.save(path)

def generate_pseudo_labels(model: nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    #run inference on the unlabeled training set to generate pseudo-masks
    #this'll return a tensor of shape [N,H,W] containing the predicted class indices
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['immage'].to(device)
            coords = batch['coord'].to(device)

            logits = model(imgs, coords)
            preds = torch.argmax(logits, dim=1) #[this is going to be shape [B, H, W]
            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)

#pseudo-label dataset
class PseudoLabelDataset(Dataset):
    def __init__(self,
                 original_dataset: MVTecLOCODataset,
                 pseudo_masks: torch.Tensor):
        self.original_dataset = original_dataset
        self.pseudo_masks = pseudo_masks

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        sample['mask'] = self.pseudo_masks[idx]

def train_phase_two(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    #load phase 1 model to generate the pseudo labels
    phase1_model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=False).to(device)
    phase1_path = Path(args.checkpoint_dir) / f"{args.category}_phase1.pth"
    phase1_model.load_state_dict(torch.load(phase1_path))

    print("generating pseudo-labels with phase 1 model")
    unlabeled_dataset = data_manager.get_train_loader().dataset
    pseudo_masks = generate_pseudo_labels(phase1_model, data_manager.get_train_loader(), device)

    #create pseudo-label dataset and accompanying dataloader
    pseudo_dataset = PseudoLabelDataset(unlabeled_dataset, pseudo_masks)
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    #set up the phase 2 model
    phase2_model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=True).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, phase2_model.parameters()), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    print(f"starting phase 2 training (pseudo-label refinement) for category {args.category}")

    phase2_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in pseudo_loader:
            optimizer.zero_grad()

            imgs = batch['image'].to(device)
            coords = batch['coord'].to(device)
            masks = batch['mask'].to(device)

            logits = phase2_model(imgs, coords)
            loss = ce_loss(logits, masks)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(pseudo_loader)
        print(f"Epoch[{epoch+1}/{args.epochs}] | Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            phase2_model.eval()
            with torch.no_grad():
                # Grab a single batch from your pseudo-label dataloader
                sample_batch = next(iter(pseudo_loader))
                sample_imgs = sample_batch['image'].to(device)
                sample_coords = sample_batch['coord'].to(device)

                sample_logits = phase2_model(sample_imgs, sample_coords)
                save_pseudo_label_samples(sample_imgs, sample_logits, epoch + 1)

            phase2_model.train()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(phase2_model.state_dict(), save_dir / f"{args.category}_phase2.pth")
    print(f"phase 2 model saved to {save_dir / f'{args.category}_phase2.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine pseudo labels (phase 2 training)")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True, help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes (background counts too)")
    parser.add_argument("--batch_size", type=int, default=8)  # Can use larger batch size here
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Where Phase 1 model is saved")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    train_phase_two(args)