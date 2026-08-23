from pathlib import Path
from typing import Tuple, List, Dict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils import pin_memory


class MVTecLOCODataset(Dataset):
    #dataset for images, masks, and coordinate grids

    def __init__(self,
                 root_dir: str,
                 category: str, #possible categories are 'good', 'logical_anomalies', and 'structural_anomalies' but this only matters for testing
                 split: str = 'train',
                 image_size: Tuple[int,int] = (512,512),
                 load_masks: bool = False,
                 few_shot:bool = False,
                 anomaly_type: str | None = None):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.load_masks = load_masks
        self.few_shot = few_shot
        self.anomaly_type = anomaly_type

        self.images: List[Path] = []
        self.masks: List[Path] = []
        self.labels: List[int] = [] #0 is normal, 1 is anomalous

        self._load_file_lists()

    def _load_file_lists(self):
        if self.few_shot:
            #only load the few-shot annotations
            images_dir: Path = self.root_dir / 'Images_fewshot_512' / self.category
            masks_dir = self.root_dir / 'Annotations_fewshot_512' / self.category

            if images_dir.exists():
                img_files = sorted(images_dir.glob('*.png')) #everything with a png file extension in the image path
                self.images = img_files
                self.masks = [masks_dir / img.name for img in img_files] #masks and their associated files are named the same
                self.labels = [0] * len(img_files) # all few-shot images are normal

        elif self.split == 'train':
            images_dir = self.root_dir / 'orig_512' / self.category / 'train' / 'good' #the training images are all good and normal

            if images_dir.exists():
                img_files = sorted(images_dir.glob('*.png'))
                self.images = img_files
                self.masks = [] # no masks in the training data
                self.labels = [0] * len(img_files) #all the images in the mvtec_loco training set are good

        elif self.split == 'validation':
            images_dir = self.root_dir / 'orig_512' / self.category / 'validation' / 'good'

            if images_dir.exists():
                img_files = sorted(images_dir.glob('*.png'))
                self.images = img_files
                self.masks = [] # no masks in the validation data
                self.labels = [0] * len(img_files)

        elif self.split == 'test':
            if self.anomaly_type is None:
                raise ValueError("anomaly_type must be specified during testing (Possible values are 'good', 'logical_anomalies', and 'structural_anomalies')")

            images_dir = self.root_dir / 'orig_512' / self.category / 'test' / self.anomaly_type
            masks_dir = self.root_dir / 'orig_512' / self.category / 'ground_truth' / self.anomaly_type

            if images_dir.exists():
                img_files = sorted(images_dir.glob('*.png'))
                self.images = img_files

                #try and load the masks if they're available
                if self.load_masks and masks_dir.exists():
                    for img_file in img_files:
                        mask_subdir = masks_dir / img_file.stem #masks are in subdirectories but have corresponding names to the images
                        if mask_subdir.exists():
                            mask_files = sorted(mask_subdir.glob('*.png'))
                            if mask_files:
                                self.masks.append(mask_files[0]) #quirk of the way they're stored in the dataset, there's only one mask per folder
                            else:
                                self.masks.append(None)
                        else:
                            self.masks.append(None)
                else:
                    self.masks = [None] * len(img_files)

                #set labels of the images
                if self.anomaly_type == 'good':
                    self.labels = [0] * len(img_files)
                else:
                    self.labels = [1] * len(img_files)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        gives a dictionary with the following keys:
            -image: the rgb image tensor [3, H, W]
            -mask: the segmentation mask tensor [H,W] (only present if testing and load_masks=True, otherwise None)
            -label: 0 or 1 for normal or anomalous respectively
            -coord: the coordinate grid tensor of shape [2, H, W]
        """

        image = Image.open(self.images[idx]).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.BILINEAR) #shouldn't need to resize because they're preprocessed but it's a safety step
        image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float()
        _, H, W = image_tensor.shape

        #load mask if available
        mask_tensor = None
        if self.load_masks:
            if self.masks[idx] is not None and self.masks[idx].exists():
                mask = Image.open(self.masks[idx])
                mask = mask.resize(self.image_size, Image.Resampling.NEAREST)
                mask_tensor = torch.from_numpy(np.array(mask)).long()
            else:
                mask_tensor = torch.zeros((H,W))

        coord_tensor = self._generate_coordinate_grid()

        result = {
            'image': image_tensor,
            'coord': coord_tensor,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if mask_tensor is not None:
            result['mask'] = mask_tensor

        return result

    def _generate_coordinate_grid(self) -> torch.Tensor:
        #just make a standard coordinate grid of shape [2,H,W]
        h, w = self.image_size
        y = torch.linspace(0, 1, h)
        x = torch.linspace(0, 1, w)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=0)

class MVTecLOCODataLoader:
    #wrapper class for the dataset class defined above
    def __init__(
            self,
            root_dir: str,
            category:str,
            batch_size: int = 8,
            num_workers: int = 4,
            image_size: Tuple[int,int] = (512,512)
    ):
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def get_train_loader(self, few_shot: bool = False) -> DataLoader:
        #normal training data dataloader
        dataset = MVTecLOCODataset(
            root_dir=self.root_dir,
            category=self.category,
            split='train',
            image_size = self.image_size,
            load_masks = False,
            few_shot=few_shot
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_few_shot_loader(self) -> DataLoader:
        dataset = MVTecLOCODataset(
            root_dir=self.root_dir,
            category=self.category,
            split='few_shot',
            image_size=self.image_size,
            load_masks=True,
            few_shot=True
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_validation_loader(self) -> DataLoader:
        dataset = MVTecLOCODataset(
            root_dir=self.root_dir,
            category=self.category,
            split='validation',
            image_size=self.image_size,
            load_masks=False
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_loader(self, anomaly_type: str) -> DataLoader:
        #get test dataloader. the anomaly_type arg refers to either 'good', 'logical_anomalies', or 'structural_anomalies'
        dataset = MVTecLOCODataset(
            root_dir=self.root_dir,
            category=self.category,
            split='test',
            image_size=self.image_size,
            load_masks=True,
            anomaly_type=anomaly_type
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_all_test_loaders(self) -> Dict[str, DataLoader]:
        #just get all the test dataloaders in a dict
        anomaly_types = ['good', 'logical_anomalies', 'structural_anomalies']
        loaders = {}
        for atype in anomaly_types:
            loaders[atype] = self.get_test_loader(anomaly_type=atype)
        return loaders


if __name__ == '__main__':
    # for testing purposes just create a breakfast box dataloader
    data_loader = MVTecLOCODataLoader(
        root_dir='MVTec_LOCO_AD_512size',
        category='breakfast_box',
        batch_size=4,
        num_workers=2
    )

    # get few-shot training data
    few_shot_loader = data_loader.get_few_shot_loader()
    print(f"Few-shot samples: {len(few_shot_loader.dataset)}")

    # get normal training data
    train_loader = data_loader.get_train_loader()
    print(f"Training samples: {len(train_loader.dataset)}")

    # get test data
    test_loaders = data_loader.get_all_test_loaders()
    for atype, loader in test_loaders.items():
        print(f"Test {atype}: {len(loader.dataset)} samples")

    # iterate through a batch
    batch = next(iter(few_shot_loader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape if 'mask' in batch else 'N/A'}")
    print(f"Coordinate grid shape: {batch['coord'].shape}")
    print(f"Labels: {batch['label']}")