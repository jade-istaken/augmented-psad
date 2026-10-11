import torch
import torch.nn as nn
import torch.nn.functional as F
from clipseg.clipseg import CLIPDensePredT
from pathlib import Path
import urllib.request
import zipfile

class Segmenter(nn.Module):
    # based on clipseg https://github.com/timojl/clipseg/tree/master with a minor modification for coordinate injection and multi-class classification

    def __init__(self, num_classes, use_coord = True, pretrained = True):
        super().__init__()
        self.use_coord = use_coord
        self.num_classes = num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPDensePredT(version='ViT-B/16', extract_layers=True)
        if pretrained:
            #have to get the pretrained weights but they're not actually part of the repo
            weight_dir = Path("clipseg_weights/")
            if not weight_dir.exists():
                weight_dir.mkdir()
            weights = weight_dir / 'rd64-uni-refined.pth'
            if not weights.exists():
                urllib.request.urlretrieve('https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download', weight_dir / 'weights.zip')
                with zipfile.ZipFile(weight_dir / 'weights.zip') as zip_ref:
                    zip_ref.extractall()

            self.model.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)

            for param in self.model.parameters():
                #freeze the backbone and decoder so that things are good for few shot training
                param.requires_grad = False

            self.multi_class_head = nn.Conv2d(16, num_classes, kernel_size=1) #by default clipseg uses a binary head
            if self.use_coord:
                self.coord_proj = nn.Conv2d(2, 16, kernel_size=1, bias=False)
            else:
                self.coord_proj = None



    def forward(self, in_tensor: torch.Tensor, coord: torch.Tensor = None):
        #clipseg, being clip-based, expects some prompts, but we can just give a list of generic prompts
        #and let the visual decoder head do the actual classification
        prompts = ['object'] * in_tensor.shape[0]

        with torch.no_grad():
            _, features = self.model(in_tensor, prompts) # we discard the logits, only need the features

        decoder_features = features[-1] # features has shape [B,16,H,W]

        if self.use_coord and coord is not None:
            target_size = decoder_features.shape[-2:] #get coord dimensions
            coord_resized = F.interpolate(coord, size=target_size, mode='bilinear', align_corners=False)
            coord_proj = self.coord_proj(coord_resized)
            decoder_features = decoder_features + coord_proj #concatenat the coordinate tensor to the features

        out = self.multi_class_head(decoder_features)

        if out.shape[-1] != 512:
            #resize as necessary
            out = F.interpolate(out, size=(512,512), mode='bilinear')


        return out
