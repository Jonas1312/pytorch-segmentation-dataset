import matplotlib.pyplot as plt

from torchvision import transforms
from utils import SegmentationDataset


def main():
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
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
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

    print(len(dataset))

    for i in range(len(dataset)):
        img_, mask_ = dataset[i]
        img_ = img_.permute(1, 2, 0)
        mask_ = mask_.permute(1, 2, 0)
        print(img_.size())
        print(mask_.size())
        plt.imshow(img_)
        plt.figure()
        plt.imshow(mask_)
        plt.show()


if __name__ == "__main__":
    main()
