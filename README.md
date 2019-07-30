# PyTorch Segmentation Dataset Loader

Custom segmentation dataset class for torchvision.

## Usage

Can be used with torchvision.transforms:

``` python
from utils import SegmentationDataset

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            resample=2,
            fillcolor=0,
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

dataset = SegmentationDataset(
    dir_images="./my_dataset/images/",
    dir_masks="./my_dataset/masks/",
    transform=transform,
)
```
