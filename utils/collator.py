import torch
from dataset.augmentation import cutmix, mixup


class MixCollator:
    def __init__(self, alpha, mode='cutmix'):
        assert mode in ['cutmix', 'mixup']
        self.alpha = alpha
        self.mode = mode

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if self.mode == 'cutmix':
            batch = cutmix(batch, self.alpha)
        elif self.mode == 'mixup':
            batch = mixup(batch, self.alpha)
        return batch
