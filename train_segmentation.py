import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.segmentation import Segmenter
from data.dataset import MVTecLOCODataLoader

#loss functions
class EntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min= 1e-8, max=1.0) #clamp to [1e-8, 1.0] in order to prevent impossible probabilities or log(0)
        entropy = -(probs * torch.log(probs)).sum(dim=1) #standard entropy calculation
        return entropy.mean()

class HistogramLoss(nn.Module):
    #loss function which compares class distribution of predictions to ground truth
    def forward(self,
                labeled_logits: torch.Tensor,
                unlabeled_logits: torch.Tensor,
                labeled_masks: torch.Tensor) -> torch.Tensor:
        labeled_probs = torch.softmax(labeled_logits, dim=1) #target histogram from labeled ground truth

        one_hot = torch.zeros_like(labeled_probs)
        one_hot.scatter_(1, labeled_masks.unsqueeze(1), 1.0) #we do a one-hot encoding of the labeeled mask to get its distribution
        target_hist = one_hot.mean(dim=[0,2,3]) #tensor of shape [num_classes

        unlabeled_probs = torch.softmax(unlabeled_logits, dim=1)
        pred_hist = unlabeled_probs.mean(dim=[0,2,3]) #create the prediction histogram

        return nn.functional.mse_loss(pred_hist, target_hist) #mse loss between the two

def train_phase_one(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initialize dataloader
    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    few_shot_loader = data_manager.get_few_shot_loader()
    unlabeled_loader = data_manager.get_train_loader()

    #initialize model
    model = Segmenter(num_classes=args.num_classes, use_coord=True, pretrained=True).to(device)

    #initialize optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    #initialize losses
    ce_loss = nn.CrossEntropyLoss()
    entropy_loss = EntropyLoss()
    hist_loss = HistogramLoss()

    print(f"Starting Phase 1 Training for category: {args.category}")

    model.train()
    for epoch in range(args.epoch):
        epoch_loss = 0.0

        unlabeled_iter = iter(unlabeled_loader) #this iterator goes through the unlabeled items

        for batch_idx, labeled_batch in enumerate(few_shot_loader):
            try:
                #this try except thing basically just makes sure that the unlabeled iterator doesn't run out of images before the few shot loader does
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            optimizer.zero_grad()

            #move all the tensor to the device
            l_imgs = labeled_batch['image'].to(device)
            l_masks = labeled_batch['mask'].to(device)
            l_coords = labeled_batch['coord'].to(device)

            u_imgs = unlabeled_batch['image'].to(device)
            u_coords = unlabeled_batch['coord'].to(device)

            #forward pass
            labeled_logits = model(l_imgs, l_coords)
            unlabeled_logits = model(u_imgs, u_coords)

            #compute losses
            loss_supervised = ce_loss(labeled_logits, l_masks)
            loss_entropy = entropy_loss(unlabeled_logits)
            loss_histogram = hist_loss(labeled_logits, unlabeled_logits, l_masks)

            total_loss = loss_supervised + 0.1 * loss_entropy + 1.0 * loss_histogram #the weighting is kinda arbitrary for now, will need to be adjusted via experimentation
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(few_shot_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.4f} | Sup: {loss_supervised.item():.4f} | Ent: {loss_entropy.item():.4f} | Hist: {loss_histogram.item():.4f}")

    #save the results of phase 1 training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f"{args.category}_phase1.pth")
    print(f"Phase 1 model saved to {save_dir / f'{args.category}_phase1.pth'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train baseline segmentation model (Phase 1)")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category", type=str, required=True, help="Category [juice_bottle, splicing_connectors, pushpins, screw_bag, breakfast_box]")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes (background counts too)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    train_phase_one(args)

    #category class numbers in mvtec_loco:
    #screw_bag = 7
    #breakfast_box = 7
    #juice_bottle = 9
    #pushpins = 26
    #splicing_connectors = 10
