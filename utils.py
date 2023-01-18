import os
import glob
import re
import torch
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(model, saved_dir, file_name='latest'):
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path+'.pt')


def increment_path(path, exist_ok=False):
    """ Automatically increment path,
        i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
