import os
import argparse
import multiprocessing
import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing
from importlib import import_module
from tqdm import tqdm
from dataset import CustomDataset, get_data
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def inference(model, test_loader, device, mode):
    model.to(device)
    model.eval()

    model_preds = []
    logits = []

    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            model_pred = model(img)
            if mode in ['logit', 'both']:
                logits.extend(model_pred.detach().cpu().numpy().tolist())
            if mode in ['answer', 'both']:
                model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
    return model_preds, logits


def parse_arg():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--model', type=str, default='BaseModel')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./output/submission/')
    parser.add_argument('--model_path', type=str, default='./output/model/exp/latest.pt')
    parser.add_argument('--mode', type=str, default='answer')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    print(args)

    if args.mode not in ['answer', 'logit', 'both']:
        raise ValueError(f'Unknown mode ({args.mode})')

    warnings.filterwarnings('ignore')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_workers = multiprocessing.cpu_count() // 2

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=50)
    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    test_df['img_path'] = test_df['img_path'].apply(
        lambda x: os.path.join(args.data_dir, x[2:]))
    test_img_paths = get_data(test_df, infer=True)
    test_transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    test_dataset = CustomDataset(test_img_paths, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers)

    preds, logits = inference(model, test_loader, device, args.mode)

    os.makedirs(args.output_dir, exist_ok=True)
    submit = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))

    if args.mode in ['answer', 'both']:
        df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
        le = preprocessing.LabelEncoder()
        df['artist'] = le.fit_transform(df['artist'].values)
        preds = le.inverse_transform(preds)
        submit['artist'] = preds
        path = os.path.join(args.output_dir, 'answer.csv')
        submit.to_csv(path, index=False)
    if args.mode in ['logit', 'both']:
        submit["artist"] = logits
        path = os.path.join(args.output_dir, 'logit.csv')
        submit.to_csv(path, index=False)
