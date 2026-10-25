import argparse
import torch
from pathlib import Path
from data.dataset import MVTecLOCODataLoader
from models.segmentation import Segmenter

def extract_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    data_manager = MVTecLOCODataLoader(
        root_dir=args.root_dir,
        category=args.category,
        batch_size=1,
        num_workers=args.num_workers
    )
    few_shot_loader = data_manager.get_few_shot_loader()

    model = Segmenter(num_classes=args.num_classes)

    support_set = {1: {'images': [], 'masks': []}}
    for batch in few_shot_loader:
        for b in range(batch['image'].shape[0]):
            mask = batch['mask'][b]
            if (mask > 0).any():
                binarized_mask = (mask > 0).float()

                support_set[1]['images'].append(batch['image'][b:b + 1])
                support_set[1]['masks'].append(binarized_mask.unsqueeze(0))

    for cls_id in support_set:
        if len(support_set[cls_id]['images']) > 0:
            support_set[cls_id]['images'] = torch.cat(support_set[cls_id]['images'], dim=0)
            support_set[cls_id]['masks'] = torch.cat(support_set[cls_id]['masks'], dim=0)

    model.support_set = support_set

    save_dir = Path(args.save_dir) / args.category
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_support_set(save_dir / "support_set.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    extract_and_save(args)