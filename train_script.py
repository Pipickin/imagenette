import os

import pandas as pd
import cv2
import numpy as np
import timm
import torch
import pytorch_lightning as pl
from torch import nn
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from timm.loss import LabelSmoothingCrossEntropy
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from src.dataset import get_dataloaders
from src.PLModel import Classifier
import src.train_utils as tu


def get_transforms(height, width):
    train_transforms_ = A.Compose([
    A.Resize(height, width),
    A.ColorJitter(
        brightness=(0.65, 1.35),
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5), 
        hue=(-0.02, 0.02),
        always_apply=True, p=1),
    A.HorizontalFlip(p=0.6),
    A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=35, p=0.5),
    A.GaussNoise(p=0.6),
    A.Normalize(mean=(0., 0., 0.), std=(1, 1, 1)),
    ToTensorV2()
    ])
    val_transforms_ = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0., 0., 0.), std=(1, 1, 1)),
        ToTensorV2()
    ])
    return train_transforms_, val_transforms_

def train(
    model_name,
    loss_name,
    loss_args,
    optimizer_name,
    scheduler_name,
    lr,
    cfg):
    train_transforms, val_transforms = get_transforms(
        cfg.height, cfg.width)
    dl_train, dl_val = get_dataloaders(
        cfg.data_path, train_transforms, 
        val_transforms, cfg.batch_size_train, 
        cfg.batch_size_val, cfg.workers)

    ## Train
    classifier = Classifier(
        model_name, 
        loss_name, 
        loss_args,
        optimizer_name, 
        scheduler_name, 
        lr=lr)

    csv_logger = tu.get_optuna_logger(
        model_name, 
        loss_name,
        loss_args,
        optimizer_name,
        scheduler_name,
        lr
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        gpus=1,
        precision=16,
        limit_train_batches=1.0, 
        limit_val_batches=1.0,
        val_check_interval=1.0,
        logger=csv_logger
    )

    trainer.fit(classifier, dl_train, dl_val)
    tu.plot_metrics(
        metrics_df_path=os.path.join(csv_logger.log_dir, 'metrics.csv'),
        save_path=os.path.join(csv_logger.log_dir)
    )
    extra_data = tu.get_extra_data(
        os.path.join(csv_logger.log_dir, 'metrics.csv'))
    best_val_acc = extra_data['val_acc']['max']
    return best_val_acc

