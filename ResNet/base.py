import os
import random
import logging
from numpy import ma
from numpy.core.fromnumeric import mean
import torch
import numpy as np
import torch.utils.data as data

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets import VisionDataset

from utils import read_images

logger = logging.getLogger(__name__)

NUM_CLASSES = 22
LABELS = [
    "ape", "bear", "bison", "cat", 
    "chicken", "cow", "deer", "dog",
    "dolphin", "duck", "eagle", "fish", 
    "horse", "lion", "lobster", "pig", 
    "rabbit", "shark", "snake", "spider", 
    "turkey", "wolf"
]
LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat", 
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish", 
    12: "horse", 13: "lion", 14: "lobster", 
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake", 
    19: "spider", 20:  "turkey", 21: "wolf"
}

class LabeledAnimalDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        train: bool,
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
        ) -> None:
            
            super().__init__(root, transform=transform, 
                             target_transform=target_transform)
            self.train = train

            self.data = []
            self.targets = []

            dir = os.path.join(root, "train")
            for i, sub_dir in enumerate(sorted(os.listdir(dir))):
                imgs = read_images(os.path.join(dir, sub_dir))
                
                if self.train:
                    imgs = imgs[: int(0.8 * len(imgs))]
                else:
                    imgs = imgs[int(0.8 * len(imgs)): ]
                
                self.data.extend(imgs)    
                self.targets.extend([i, ] * len(imgs))

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

class UnlabeledAnimalDataset(VisionDataset):
    """[summary]

    Args:
        VisionDataset ([type]): [description]
    """

    def __init__(self, 
                root: str, 
                transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

from torchvision import transforms

if __name__ == '__main__':
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
    ])
    
    labeled_animal_dataset = LabeledAnimalDataset("data", train=False, 
                                                  transform=transform_labeled)

    print(f"len: {len(labeled_animal_dataset)}")

    dataloader = torch.utils.data.DataLoader(
        labeled_animal_dataset, 
        batch_size=len(labeled_animal_dataset),
        shuffle=True
    )

    for images, labels in dataloader:
        arrs = images.numpy()

        # x = np.mean(arrs, axis=0)
        # x = np.mean(x, axis=1)
        # x = np.mean(x, axis=1)
        # print(x)
        
        means = []
        stds = []

        for map in np.transpose(arrs, (1, 0, 2, 3)):
            # assert(map.shape == (, 32, 32))
            
            means.append(np.mean(map))
            stds.append(np.std(map))
        
        print(means)
        print(stds)
        # np.std