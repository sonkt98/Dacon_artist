import os
import argparse
import warnings
import multiprocessing
import numpy as np
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import get_dataset
from utils.collator import MixCollator
from utils.criterion import create_criterion, MixCriterion
from utils.fold import train_kfold
from utils.metric import validation
from utils.scheduler import get_scheduler
from utils.util import seed_everything, save_model, increment_path


def train(model, optimizer, train_loader, test_loader, scheduler,
          device, saved_dir, args):

    model.to(device)

    criterion = create_criterion(args.criterion).to(device)
    if args.cutmix or args.mixup:
        criterion = MixCriterion(criterion)
    if not args.no_valid:
        val_criterion = create_criterion(args.criterion).to(device)

    best_score = 0
    patience = args.early_stopping if args.early_stopping > 0 else 9999

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = []

        for img, label in tqdm(iter(train_loader)):

            img = img.float().to(device)

            if args.cutmix or args.mixup:
                targets1, targets2, lam = label
                label = (targets1.to(device), targets2.to(device), lam)
            else:
                label = label.to(device)

            optimizer.zero_grad()

            model_pred = model(img)

            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)

        if args.no_valid:
            print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}]')
        else:
            val_loss, val_score = validation(model, val_criterion, test_loader, device)
            print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')

            if best_score < val_score:
                best_score = val_score
                file_name = f'{args.model}_Epoch_{epoch}_F1_{best_score:.5f}'
                save_model(model, saved_dir, file_name)
                if args.early_stopping > 0:
                    patience = args.early_stopping
            elif args.early_stopping > 0:
                patience -= 1

        if args.early_stopping > 0 and patience < 1:
            print('Early stopping ...')
            break

        if scheduler is not None:
            scheduler.step()

        if epoch == args.epochs:
            save_model(model, saved_dir)


def parse_arg():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--model', type=str, default='BaseModel')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--criterion', type=str, default='cross_entropy')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation')
    parser.add_argument('--resize', type=int, default=480)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--name', type=str, default='exp', help='model save at {name}')
    parser.add_argument('--no_valid', action='store_true')

    # KFold arguments
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--n_splits', type=int, default=7)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--oof', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arg()
    print(args)

    warnings.filterwarnings('ignore')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_dir = increment_path(os.path.join('./output/model', args.name))

    seed_everything(args.seed)

    num_workers = multiprocessing.cpu_count() // 2

    if args.cutmix:
        collate_fn = MixCollator(alpha=args.alpha, mode='cutmix')
    elif args.mixup:
        collate_fn = MixCollator(alpha=args.alpha, mode='mixup')
    else:
        collate_fn = None

    if args.kfold:
        train_kfold(device, saved_dir, num_workers, collate_fn, args)
    else:
        # Dataset
        train_dataset, val_dataset = get_dataset(args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
        if args.no_valid:
            val_loader = None
        else:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=num_workers)

        # Train model
        model_module = getattr(import_module("models.model"), args.model)
        model = model_module(num_classes=50)
        model.eval()
        optimizer_module = getattr(import_module('torch.optim'), args.optimizer)
        optimizer = optimizer_module(model.parameters(), lr=args.lr)
        scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)
        train(model, optimizer, train_loader, val_loader, scheduler,
              device, saved_dir, args)
