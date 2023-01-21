import os
import argparse
import multiprocessing
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from dataset import get_dataset
from utils import seed_everything, competition_metric, \
                save_model, increment_path
from augmentation import MixCollator, MixCriterion
from loss import create_criterion
from scheduler import get_scheduler


def train(model, optimizer, train_loader, test_loader, scheduler,
          device, saved_dir, args):

    model.to(device)

    criterion = create_criterion(args.criterion).to(device)
    if args.use_cutmix or args.use_mixup:
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

            if args.use_cutmix or args.use_mixup:
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


def validation(model, criterion, test_loader, device):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1


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
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--name', type=str, default='exp', help='model save at {name}')
    parser.add_argument('--no_valid', action='store_true')
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

    if args.use_cutmix:
        collate_fn = MixCollator(alpha=args.alpha, mode='cutmix')
    elif args.use_mixup:
        collate_fn = MixCollator(alpha=args.alpha, mode='mixup')
    else:
        collate_fn = None

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
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=50)
    model.eval()
    optimizer_module = getattr(import_module('torch.optim'), args.optimizer)
    optimizer = optimizer_module(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)
    train(model, optimizer, train_loader, val_loader, scheduler,
          device, saved_dir, args)
