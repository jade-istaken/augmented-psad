import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import numpy as np

from models.segmentation import Segmenter
from data.dataset import MVTecLOCODataLoader, MVTecLOCODataset

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