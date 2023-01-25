import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from importlib import import_module
from dataset.dataset import get_data, CustomDataset
from utils.criterion import create_criterion, MixCriterion
from utils.scheduler import get_scheduler
from utils.metric import validation
from utils.util import save_model


def getDataloader(train_dataset, val_dataset,
                  train_idx, valid_idx, batch_size, num_workers,
                  collator):
    train_set = torch.utils.data.Subset(train_dataset,
                                        indices=train_idx)
    val_set = torch.utils.data.Subset(val_dataset,
                                      indices=valid_idx)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collator)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader


def train_kfold(device, saved_dir, num_workers, collate_fn, args):

    if args.stratified:
        kf = StratifiedKFold(n_splits=args.n_splits)
    else:
        kf = KFold(n_splits=args.n_splits)

    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df['img_path'] = df['img_path'].apply(
        lambda x: os.path.join(args.data_dir, x[2:]))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    all_img_paths, all_labels = get_data(df)

    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    test_df['img_path'] = test_df['img_path'].apply(
        lambda x: os.path.join(args.data_dir, x[2:]))
    test_img_paths = get_data(test_df, infer=True)
    test_transform_module = getattr(import_module('dataset.augmentation'), 'TestAugmentation')
    test_transform = test_transform_module(args.crop_size)
    test_dataset = CustomDataset(test_img_paths, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers)

    criterion = create_criterion(args.criterion).to(device)
    if args.cutmix or args.mixup:
        criterion = MixCriterion(criterion)
    val_criterion = create_criterion(args.criterion).to(device)

    train_transform_module = getattr(import_module('dataset.augmentation'), args.augmentation)
    train_transform = train_transform_module(args.resize, args.crop_size)
    train_dataset = CustomDataset(all_img_paths, all_labels, train_transform)
    val_transform_module = getattr(import_module('dataset.augmentation'), 'BaseAugmentation')
    val_transform = val_transform_module(args.resize, args.crop_size)
    val_dataset = CustomDataset(all_img_paths, all_labels, val_transform)

    oof_pred = None

    for i, (train_idx, valid_idx) in enumerate(kf.split(df.img_path.to_list(), df.artist.to_list())):

        print(f'Fold {i + 1}/{args.n_splits} ...')

        train_loader, val_loader = getDataloader(
            train_dataset, val_dataset, train_idx, valid_idx,
            args.batch_size, num_workers, collate_fn
        )

        model_module = getattr(import_module("models.model"), args.model)
        model = model_module(num_classes=50)
        model.to(device)

        optimizer_module = getattr(import_module('torch.optim'), args.optimizer)
        optimizer = optimizer_module(model.parameters(), lr=args.lr)
        scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)

        best_score = 0
        patience = args.early_stopping if args.early_stopping > 0 else 9999
        best_model = None

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

            val_loss, val_score = validation(model, val_criterion, val_loader, device)
            print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')

            if best_score < val_score:
                best_model = model
                best_score = val_score
                file_name = f'Fold{i + 1}_{args.model}_Epoch_{epoch}_F1_{best_score:.5f}'
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

        all_predictions = []

        with torch.no_grad():
            for images in test_loader:

                images = images.float().to(device)

                if args.tta:
                    pred = best_model(images) / 2
                    pred += best_model(torch.flip(images, dims=(-1,))) / 2
                else:
                    pred = best_model(images)

                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        if args.oof:
            if oof_pred is None:
                oof_pred = fold_pred / args.n_splits
            else:
                oof_pred += fold_pred / args.n_splits

    if args.oof:
        submit = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
        oof_pred = torch.from_numpy(oof_pred)
        output_path = './output/submission/'
        os.makedirs(output_path, exist_ok=True)

        oof_pred_ans = oof_pred.argmax(dim=-1)
        oof_pred_ans = oof_pred_ans.detach().cpu().numpy().tolist()
        preds = le.inverse_transform(oof_pred_ans)
        submit['artist'] = preds
        prefix = 'oof'
        answer_path = os.path.join(output_path, prefix + '_answer.csv')
        submit.to_csv(answer_path, index=False)

        oof_pred = oof_pred.detach().cpu().numpy().tolist()
        submit['artist'] = oof_pred
        logit_path = os.path.join(output_path, prefix + '_logit.csv')
        submit.to_csv(logit_path, index=False)
