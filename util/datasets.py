import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.model_selection import StratifiedShuffleSplit

# Patient-wise Greedy Splitter
def patient_data_splitter(csv_path, test_size, seed, n_splits:int = 5):
    full_df = pd.read_csv(csv_path)
    patient_stats = full_df.groupby('patient_id').agg(
        label=('binary_label', 'first'),
        img_count=('file_path', 'count')
    ).reset_index()
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss.split(patient_stats, patient_stats['label']))
    
    test_pids = patient_stats.iloc[test_idx]['patient_id']
    test_df = full_df[full_df['patient_id'].isin(test_pids)].reset_index(drop=True)
    
    trainval_stats = patient_stats.iloc[trainval_idx].copy()
    df_rem = full_df[full_df['patient_id'].isin(trainval_stats['patient_id'])].reset_index(drop=True)
    
    pos_stats = trainval_stats[trainval_stats['label'] == 1].sort_values(by='img_count', ascending=False)
    neg_stats = trainval_stats[trainval_stats['label'] == 0].sort_values(by='img_count', ascending=False)

    folds_pids = [[] for _ in range(n_splits)]
    folds_img_counts = np.zeros(n_splits)

    for pool in [pos_stats, neg_stats]:
        for _, row in pool.iterrows():
            idx = np.argmin(folds_img_counts)
            folds_pids[idx].append(row['patient_id'])
            folds_img_counts[idx] += row['img_count']

    folds = []
    for i in range(n_splits):
        val_pids = folds_pids[i]
        # Get indices relative to df_rem
        val_idx = df_rem[df_rem['patient_id'].isin(val_pids)].index.values
        train_idx = df_rem[~df_rem['patient_id'].isin(val_pids)].index.values
        folds.append((train_idx, val_idx))

    return df_rem, test_df, folds

class ImageDFDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Assumes 'file_path' is absolute or relative to execution dir
        img = Image.open(row['file_path']).convert('RGB')
        label = int(row['binary_label'])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

# ---- helpers ----

def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep

