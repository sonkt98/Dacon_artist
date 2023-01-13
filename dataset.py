import os
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values


def get_dataset(args):
    path = os.path.join(args.data_dir, 'train.csv')
    df = pd.read_csv(path)
    df['img_path'] = df['img_path'].apply(
        lambda x: os.path.join(args.data_dir, x[2:]))
    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)

    train_df, val_df, _, _ = train_test_split(df,
                                              df['artist'].values,
                                              test_size=0.2,
                                              random_state=41)
    train_df = train_df.sort_values(by=['id'])
    val_df = val_df.sort_values(by=['id'])

    train_img_paths, train_labels = get_data(train_df)
    val_img_paths, val_labels = get_data(val_df)

    train_transform = A.Compose([
                        A.Resize(480, 480),
                        A.RandomCrop(224, 224),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0,
                            always_apply=False,
                            p=1.0),
                        ToTensorV2()
                        ])

    val_transform = A.Compose([
                        A.Resize(480, 480),
                        A.RandomCrop(224, 224),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0,
                            always_apply=False,
                            p=1.0),
                        ToTensorV2()
                        ])

    train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
    val_dataset = CustomDataset(val_img_paths, val_labels, val_transform)
    return train_dataset, val_dataset
