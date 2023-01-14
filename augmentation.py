import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BaseAugmentation:
    def __init__(self, resize, crop_size, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(crop_size, crop_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=np.array(image))


class AugmentationV1:
    def __init__(self, resize, crop_size, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=np.array(image))


class AugmentationV2:
    def __init__(self, resize, crop_size, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(crop_size, crop_size),
            A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=np.array(image))


class AugmentationV3:
    def __init__(self, resize, crop_size, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(crop_size, crop_size),
            A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=np.array(image))
