import os
import glob
import argparse
import warnings
import json
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn import preprocessing


def voting(path, label_encoder, method='soft'):
    assert method in ['soft', 'hard']
    files = glob.glob(path)
    preds = []

    if method == 'soft':
        softmax = nn.Softmax(dim=1)
        for file in files:
            df = pd.read_csv(file, index_col=None, header=0)
            if len(preds) == 0:
                preds = df['artist']
                for i in range(len(preds)):
                    x = [json.loads(preds[i])]
                    x = softmax(torch.tensor(x))
                    preds[i] = x.tolist()
            else:
                for i in range(len(preds)):
                    x = torch.tensor(preds[i])
                    y = [json.loads(df['artist'][i])]
                    y = softmax(torch.tensor(y))
                    preds[i] = (x.add(y)).tolist()
        preds = torch.tensor(preds).argmax(dim=-1)

    elif method == 'hard':
        preds_list = []
        for file in files:
            df = pd.read_csv(file, index_col=None, header=0)
            df['artist'] = label_encoder.transform(df['artist'].values)
            if len(preds_list) == 0:
                for i in df['artist'].tolist():
                    preds_list.append([i])
            else:
                for i in range(len(preds_list)):
                    preds_list[i].append(df['artist'][i])
        for i in range(len(preds_list)):
            dic = Counter(preds_list[i])
            preds.append(dic.most_common()[0][0])

    preds = le.inverse_transform(preds)
    return preds


def parse_arg():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ensemble_path', type=str, default='./output/ensemble/')
    parser.add_argument('--output_dir', type=str, default='./output/submission/')
    parser.add_argument('--voting', type=str, default='soft')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    print(args)

    if args.voting not in ['soft', 'hard']:
        raise ValueError(f'Unknown voting ({args.mode})')

    warnings.filterwarnings('ignore')

    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)

    preds = voting(os.path.join(args.ensemble_path, '*.csv'), le,
                   method=args.voting)

    os.makedirs(args.output_dir, exist_ok=True)
    submit = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    submit['artist'] = preds
    path = os.path.join(args.output_dir, 'ensemble_' + args.voting + '.csv')
    submit.to_csv(path, index=False)
