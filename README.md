# PyTorch Segmentation Dataset Loader

Custom segmentation dataset class for `torchvision`.

## Usage

Can be used with `torchvision.transforms`:

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

## Note

- `Normalize`, `Lambda`, `Pad`, `ColorJitter` and `RandomErasing` won't be applied to masks by default
- Images from: https://www.ntu.edu.sg/home/asjfcai/Benchmark_Website/benchmark_index.html

## Helpful Links

- https://github.com/sshuair/torchvision-enhance
- https://github.com/jbohnslav/opencv_transforms
- https://github.com/aleju/imgaug
- https://github.com/albu/albumentations
