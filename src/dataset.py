import os
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import albumentations as A


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

    
def get_dataloaders(
    data_path, 
    train_transforms,
    val_transforms,
    batch_size_train,
    batch_size_val,
    num_workers=32) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'train'), 
        transform=Transforms(transforms=train_transforms))
    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'), 
        transform=Transforms(transforms=val_transforms))
    
    dl_train = DataLoader(
        train_dataset, batch_size=batch_size_train, 
        shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(
        val_dataset, batch_size=batch_size_val,
        shuffle=False, num_workers=num_workers)
    return dl_train, dl_val