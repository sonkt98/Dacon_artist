import numpy as np
import torch
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


class TestAugmentation:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize),
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


def cutmix(batch, alpha=1.0):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


def mixup(batch, alpha=0.2):
    data, targets = batch
    indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)

    data = lam * data + (1 - lam) * shuffled_data
    targets = (targets, shuffled_targets, lam)

    return data, targets
