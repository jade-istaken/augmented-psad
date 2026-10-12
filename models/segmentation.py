from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

class Segmenter(nn.Module):
    #this whole thing's SegGPT based now this is going to work WAY better no more weird multi phase training and bespoke models we're cashing in on proven solutions
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        model_id = "BAAI/seggpt-vit-large"
        self.processor = SegGptImageProcessor.from_pretrained(model_id)
        self.model = SegGptForImageSegmentation.from_pretrained(model_id).to(self.device)

        self.model.eval()
        for param in self.model.parameters():
            #this is needed to keep the model frozen, it's an in-context learning model
            param.requires_grad = False

        self.support_set = None

    def set_support_set(self, support_set: dict):
        #inject the few-shot support set from the few shot data to the model
        #dict should be like {class_id: int : {'images': [K, 3, H, W], 'masks': [K, H, W]}}

        self.support_set = {}
        for cls_id, data in support_set.items():
            if cls_id == 0:
                continue #skips it when the class is background cause like. background doesn't get a mask

            images = data['images'].cpu()
            masks = (data['masks'] == cls_id).float().cpu()
            #move them to the cpu for numpy work

    def save_support_set(self, save_path: str):
        #save the support set to a .pt
        if self.support_set is None:
            raise RuntimeError("no support set to save")

        cpu_support_set = {}
        for cls_id, data in self.support_set.items():
            cpu_support_set[cls_id] = {
                'images': data['images'].cpu(),
                'masks': data['masks'].cpu()
            }
        torch.save(cpu_support_set, save_path)
        print(f"Support set saved to {save_path}")

    def load_support_set(self, load_path: str, device: torch.device):
        #load support set from .pt
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Support set not found at {load_path}")

        cpu_support_set = torch.load(load_path, map_location='cpu')

        # move to target device
        self.support_set = {}
        for cls_id, data in cpu_support_set.items():
            self.support_set[cls_id] = {
                'images': data['images'].to(device),
                'masks': data['masks'].to(device)
            }
        print(f"Support set loaded from {load_path}")

    def forward(self, in_tensor: torch.Tensor):
        #in_tensor should be [B,3,512,512]
        B,_,H,W = in_tensor.shape
        assert H == 512 and W == 512 #just make it so that like. it haaaaas to be 512x512

        logits = torch.zeros(B, self.num_classes, H, W, device = self.device) #initialize the logits tensor to all zeros

        CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device) #needed for denormalization
        CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        denorm_imgs = (in_tensor.float() * CLIP_STD + CLIP_MEAN).clamp(0, 1)
        query_imgs = denorm_imgs.cpu().numpy()

        if self.support_set is None:
            raise RuntimeError("no support_set yet, please set_support_set() first")

        for cls_id in range (1, self.num_classes):
            if cls_id not in self.support_set:
                continue

            support_data = self.support_set[cls_id]
            support_imgs = support_data['images'] # [K,3,H,W]
            support_masks = support_data['masks'] # [K,H,W]
            K = support_imgs.shape[0]

            batch_preds = []
            for batch in range(B):
                query_image = query_imgs[batch].transpose(1,2,0) #convert CHWY to HWC
                pred_accumulator = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.float32)

                for k in range(K):
                    prompt_imgs = support_imgs[k].cpu().numpy().transpose(1, 2, 0)
                    prompt_masks = support_masks[k].cpu().numpy()

                    inputs = self.processor(
                        images = query_image,
                        prompt_images =prompt_imgs,
                        prompt_masks = prompt_masks,
                        return_tensors = 'pt'
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    pred_mask = outputs.pred_masks
                    if pred_mask.shape[1] == 3:
                        pred_mask = pred_mask[:, 0:1, :, :]
                    pred_mask = F.interpolate(pred_mask, size=(512, 512), mode='bilinear', align_corners=False)
                    pred_accumulator += pred_mask
                avg_pred = pred_accumulator / K
                batch_preds.append(avg_pred)

            cls_preds = torch.cat(batch_preds, dim=0)
            #scale predictions so they're. kind of like logits ?
            logits[:, cls_id:cls_id + 1] = cls_preds * 10.0

        return logits