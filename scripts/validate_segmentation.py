import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).parent.parent)) #ensurews project root is in path

from models.segmentation import Segmenter


#helpers
def generate_coordinate_grid(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    #generates a 2d coordinate grid normalized to [0,1] for the segmentation network
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    # shape: [2, H, W] -> [1, 2, H, W] -> [B, 2, H, W]
    coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return coords


def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # Ignore classes not present in ground truth
        else:
            ious.append(float(intersection) / union)

    # filter out the empty classes and calculate the mean
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def visualize_predictions(
        images: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        save_path: str
) -> None:
    #save the grid comparing ground truth and predictions
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    if batch_size == 1:
        axes = axes.unsqueeze(0)

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize for display

        gt = gt_masks[i].cpu().numpy()
        pred = pred_masks[i].cpu().numpy()

        # Use a distinct colormap for segmentation classes
        cmap = plt.get_cmap("tab20", 10)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=9)
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualizations saved to: {save_path}")



def main():
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # background + components
    batch_size = 4
    img_size = 256
    num_epochs = 50

    print(f"Using device: {device}")

    #prepare mock data
    print("Generating visual mock data...")

    mock_images = torch.rand(batch_size, 3, img_size, img_size).to(device) * 0.2  # Dark background noise
    mock_masks = torch.zeros(batch_size, img_size, img_size, dtype=torch.long).to(device)

    for i in range(batch_size):
        # draw orange square
        mock_images[i, 0, 50:150, 50:150] = 1.0  # R
        mock_images[i, 1, 50:150, 50:150] = 0.5  # G
        mock_images[i, 2, 50:150, 50:150] = 0.0  # B
        mock_masks[i, 50:150, 50:150] = 1

        # draw red square
        mock_images[i, 0, 160:200, 160:200] = 1.0  # R
        mock_images[i, 1, 160:200, 160:200] = 0.0  # G
        mock_images[i, 2, 160:200, 160:200] = 0.0  # B
        mock_masks[i, 160:200, 160:200] = 2

        # draw green square
        mock_images[i, 0, 50:100, 160:200] = 0.0  # R
        mock_images[i, 1, 50:100, 160:200] = 1.0  # G
        mock_images[i, 2, 50:100, 160:200] = 0.0  # B
        mock_masks[i, 50:100, 160:200] = 3

    dataset = TensorDataset(mock_images, mock_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    print("Initializing Segmenter...")
    model = Segmenter(num_classes=num_classes, use_coord=True, pretrained=True).to(device) #initialize the segmenter

    # verify that the parameters are like. good.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

    # tiny training loop to confirm gradient flow
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print("\nStarting brief training loop to verify learning...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in dataloader:
            optimizer.zero_grad()

            # generate coordinate grid
            coords = generate_coordinate_grid(images.shape[0], img_size, img_size, device)

            #forward pass
            logits = model(images, coords)  # Shape: [B, num_classes, H, W]

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f}")

    #inference time
    print("\nRunning inference on validation batch...")
    model.eval()
    with torch.no_grad():
        val_images, val_masks = next(iter(dataloader))
        val_coords = generate_coordinate_grid(val_images.shape[0], img_size, img_size, device)

        val_logits = model(val_images, val_coords)

        # get predicted masks by doing argmax on the class dimensions
        val_preds = torch.argmax(val_logits, dim=1)

        miou = calculate_miou(val_preds, val_masks, num_classes)
        print(f"Validation mIoU: {miou:.4f}")

    output_dir = Path("outputs/segmentation_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_predictions(
        images=val_images,
        gt_masks=val_masks,
        pred_masks=val_preds,
        save_path=str(output_dir / "model_predictions.png")
    )


if __name__ == "__main__":
    main()