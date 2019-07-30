import os
import random

from torch.utils import data

from torchvision import transforms
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)
from torchvision.transforms import functional as F


class SegmentationDataset(data.Dataset):
    def __init__(self, dir_images, dir_masks, transform=None, extensions=None):
        """A dataloader for segmentation datasets

        Args:
            dir_images (string): images directory path
            dir_masks (string): masks directory path
            transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version. Defaults to None.
            extensions (list, optional): A list of allowed extensions. Defaults to None.

        Note:
            Normalize, Lambda, Pad, ColorJitter and RandomErasing won't be applied to masks by default
        """
        super().__init__()
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self.transform = transform
        self.extensions = extensions if extensions else IMG_EXTENSIONS
        self.extensions = tuple(x.lower() for x in self.extensions)

        self.img_names, self.mask_names = self.make_dataset()

    def __getitem__(self, index):
        img_name = self.img_names[index]
        mask_name = self.mask_names[index]
        img = default_loader(os.path.join(self.dir_images, img_name))
        mask = default_loader(os.path.join(self.dir_masks, mask_name))

        assert img.size == mask.size

        if not self.transform:
            return img, mask

        img, mask = self.apply_transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def make_dataset(self):
        img_names = sorted(os.listdir(self.dir_images))
        mask_names = sorted(os.listdir(self.dir_masks))

        img_names = [
            x for x in img_names if has_file_allowed_extension(x, self.extensions)
        ]
        mask_names = [
            x for x in mask_names if has_file_allowed_extension(x, self.extensions)
        ]

        assert len(img_names) == len(mask_names)

        return img_names, mask_names

    def apply_transform(self, img, mask, current_transform=None):
        if current_transform is None:
            current_transform = self.transform

        if isinstance(current_transform, (transforms.Compose)):
            for transform in current_transform.transforms:
                img, mask = self.apply_transform(img, mask, transform)

        elif isinstance(current_transform, (transforms.RandomApply)):
            if current_transform.p >= random.random():
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms
                )

        elif isinstance(current_transform, (transforms.RandomChoice)):
            t = random.choice(current_transform.transforms)
            img, mask = self.apply_transform(img, mask, t)

        elif isinstance(current_transform, (transforms.RandomOrder)):
            order = list(range(len(current_transform.transforms)))
            random.shuffle(order)
            for i in order:
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms[i]
                )

        elif isinstance(
            current_transform,
            (
                transforms.CenterCrop,
                transforms.FiveCrop,
                transforms.TenCrop,
                transforms.ToTensor,
                transforms.Grayscale,
                transforms.Resize,
            ),
        ):
            img = current_transform(img)
            mask = current_transform(mask)

        elif isinstance(
            current_transform, (transforms.Normalize, transforms.Lambda, transforms.Pad)
        ):
            img = current_transform(img)
            # mask = current_transform(mask)  # apply on input only

        elif isinstance(current_transform, (transforms.ColorJitter)):
            transform = current_transform.get_params(
                current_transform.brightness,
                current_transform.contrast,
                current_transform.saturation,
                current_transform.hue,
            )
            for lambda_transform in transform.transforms:
                img = lambda_transform(img)

        elif isinstance(current_transform, (transforms.RandomAffine)):
            ret = current_transform.get_params(
                current_transform.degrees,
                current_transform.translate,
                current_transform.scale,
                current_transform.shear,
                img.size,
            )
            img = F.affine(
                img,
                *ret,
                resample=current_transform.resample,
                fillcolor=current_transform.fillcolor,
            )
            mask = F.affine(
                mask,
                *ret,
                resample=current_transform.resample,
                fillcolor=current_transform.fillcolor,
            )

        elif isinstance(current_transform, (transforms.RandomCrop)):
            i, j, h, w = current_transform.get_params(img, current_transform.size)
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        elif isinstance(current_transform, (transforms.RandomHorizontalFlip)):
            if random.random() < current_transform.p:
                img = F.hflip(img)
                mask = F.hflip(mask)

        elif isinstance(current_transform, (transforms.RandomVerticalFlip)):
            if random.random() < current_transform.p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        elif isinstance(current_transform, (transforms.RandomPerspective)):
            if random.random() < current_transform.p:
                width, height = img.size
                startpoints, endpoints = current_transform.get_params(
                    width, height, current_transform.distortion_scale
                )
                img = F.perspective(
                    img, startpoints, endpoints, current_transform.interpolation
                )
                mask = F.perspective(
                    mask, startpoints, endpoints, current_transform.interpolation
                )

        elif isinstance(current_transform, (transforms.RandomResizedCrop)):
            ret = current_transform.get_params(
                img, current_transform.scale, current_transform.ratio
            )
            img = F.resized_crop(
                img, *ret, current_transform.size, current_transform.interpolation
            )
            mask = F.resized_crop(
                mask, *ret, current_transform.size, current_transform.interpolation
            )

        elif isinstance(current_transform, (transforms.RandomRotation)):
            angle = current_transform.get_params(current_transform.degrees)

            img = F.rotate(
                img,
                angle,
                current_transform.resample,
                current_transform.expand,
                current_transform.center,
            )
            mask = F.rotate(
                mask,
                angle,
                current_transform.resample,
                current_transform.expand,
                current_transform.center,
            )

        elif isinstance(current_transform, (transforms.RandomErasing)):
            if random.uniform(0, 1) < current_transform.p:
                x, y, h, w, v = current_transform.get_params(
                    img,
                    scale=current_transform.scale,
                    ratio=current_transform.ratio,
                    value=current_transform.value,
                )
                img = F.erase(img, x, y, h, w, v, current_transform.inplace)
                # mask =  F.erase(mask, x, y, h, w, v, current_transform.inplace)

        else:
            raise NotImplementedError(
                f'Transform "{current_transform}" not implemented yet'
            )
        return img, mask
