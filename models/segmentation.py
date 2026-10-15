from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import SegGptForImageSegmentation, SegGptImageProcessor
from PIL import Image
from transformers.models.glm5_next.modeling_glm5_next import chunk_kimi_delta_attention


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
            masks = (data['masks']).float().cpu()
            #move them to the cpu for numpy work
            self.support_set[cls_id] = {
                'images': images,
                'masks': masks
            }

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
        logits[:, 0, :, :] = 1.1

        CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device) #needed for denormalization
        CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        denorm_imgs = (in_tensor.float() * CLIP_STD + CLIP_MEAN).clamp(0, 1)
        query_imgs = [
            Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            for img in denorm_imgs
        ] #batch process all at once

        if self.support_set is None:
            raise RuntimeError("no support_set yet, please set_support_set() first")

        for cls_id in range (1, self.num_classes):
            if cls_id not in self.support_set:
                continue

            support_data = self.support_set[cls_id]
            support_imgs = support_data['images'] # [K,3,H,W]
            support_masks = support_data['masks'] # [K,H,W]
            K = support_imgs.shape[0]

            #convert to PIL images all at once
            support_imgs_pil = [
                Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                for img in support_imgs
            ]
            support_masks_pil = [
                Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                for mask in support_masks
            ]

            batch_query_imgs = [query_imgs[b] for b in range(B) for _ in range(K)]
            batch_prompt_imgs = [img for _ in range(B) for img in support_imgs_pil]
            batch_prompt_masks = [mask for _ in range(B) for mask in support_masks_pil]

            chunk_size = 16  #chunking to avoid ram overflow
            total_pairs = B * K
            all_preds = []

            for start_idx in range(0, total_pairs, chunk_size):
                end_idx = min(start_idx + chunk_size, total_pairs)

                inputs = self.processor(
                    images=batch_query_imgs[start_idx:end_idx],
                    prompt_images=batch_prompt_imgs[start_idx:end_idx],
                    prompt_masks=batch_prompt_masks[start_idx:end_idx],
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                pred_mask = outputs.pred_masks
                if pred_mask.shape[1] == 3:
                    pred_mask = pred_mask[:, 0:1, :, :]

                pred_mask = torch.sigmoid(pred_mask)    #GUESS WHO FOROGT AN ACTIVATION FUNCTION

                query_h = pred_mask.shape[2] // 2 #apparently segGPT will like, stack the query mask and prompt mask by default, this fixes that
                pred_mask = pred_mask[:, :, query_h:, :]

                pred_mask = F.interpolate(pred_mask, size=(512, 512), mode='bilinear', align_corners=False)
                all_preds.append(pred_mask)

            full_pred_masks = torch.cat(all_preds, dim=0) #concatenate chunks to [B*K,1,512,512]
            cls_preds = full_pred_masks.view(B,K,1,512,512).mean(dim=1)

            #scale predictions so they're. kind of like logits ?
            logits[:, cls_id:cls_id + 1] = cls_preds * 10.0

        return logits