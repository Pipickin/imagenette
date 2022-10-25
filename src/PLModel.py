from functools import partial

import torch.optim as optim
import torch
import pytorch_lightning as pl
from torch import nn
import timm
from timm.loss import LabelSmoothingCrossEntropy
from fastai.vision.models.xresnet import *
from fastai.basics import *
from vit_pytorch import ViT




class CustomModel(nn.Module):
    def __init__(
        self,
        channels=(3, 16, 16, 32, 64),
        kernel_size=(5, 5, 5, 3),
        batch_norm=True,
        dropout_p=0,
        max_pool_kernel=5,
    ):
        super().__init__()
        self.fe = nn.Sequential()

        for i in range(len(channels) - 2):
            self.fe.append(
                nn.Conv2d(channels[i], channels[i + 1],
                          kernel_size[i],
                          stride=1))
            if batch_norm:
                self.fe.append(nn.BatchNorm2d(channels[i + 1]))
            self.fe.append(nn.ReLU())
            if dropout_p:
                self.fe.append(nn.Dropout(p=dropout_p))
        self.fe.append(nn.Conv2d(channels[-2], channels[-1],
                                 kernel_size[-1],
                                 stride=1))
        if batch_norm:
            self.fe.append(nn.BatchNorm2d(channels[-1]))
        if max_pool_kernel:
            self.fe.append(nn.MaxPool2d(max_pool_kernel))
        self.flatten = nn.Flatten()
        self.input_ln = nn.Linear(30976, 1024)
        self.relu = nn.ReLU()
        self.output_ln = nn.Linear(1024, 10) 
        
    def forward(self, x):
        x = self.fe(x)
        x = self.flatten(x)
        x = self.input_ln(x)
        x = self.relu(x)
        x = self.output_ln(x)
        return x


def model_factory(model_name):
    # Resnet
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', num_classes=10, pretrained=False)
    elif model_name == 'resnet18':
        model = timm.create_model('resnet18', num_classes=10, pretrained=False)
    elif model_name == 'xresnet50':
        model = xse_resnext50(
            n_out=10, act_cls=Mish, sa=1, sym=0, pool=MaxPool)
    # VIT
    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', num_classes=10, pretrained=False)
    elif model_name == 'convit_base':
        model = timm.create_model('convit_base', num_classes=10, pretrained=False)
    elif model_name == 'convit_tiny':
        model = timm.create_model('convit_tiny', num_classes=10, pretrained=False)
    elif model_name == 'vit_small_resnet26d_224':
        model = timm.create_model('vit_small_resnet26d_224', num_classes=10, pretrained=False)
    elif model_name == 'vit':
        model = ViT(
            image_size = 128,
            # image_size = 224,
            patch_size = 32,
            num_classes = 10,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    # custom model
    elif model_name == 'custom_batch':
        model = CustomModel()
    elif model_name == 'custom_dropout':
        model = CustomModel(
            batch_norm=False,
            dropout_p=0.3)
    elif model_name == 'custom_full':
        model = CustomModel(
            dropout_p=0.3)
    else:
        raise Exception(f'Wrong model name: {model_name}')
    return model


def loss_factory(loss_name, *args):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'smooth_cross_entropy':
        return LabelSmoothingCrossEntropy(*args)
    else:
        raise Exception(f'Wrong loss name: {loss_name}')
        
        
def optimizer_factory(optimizer_name):
    if optimizer_name == 'radam':
        return optim.RAdam
    elif optimizer_name == 'adam':
        return optim.Adam
    elif optimizer_name == 'adamw':
        return optim.AdamW
    elif optimizer_name == 'SGD':
        return optim.SGD
    else:
        raise Exception(f'Wrong optimizer name: {optimizer_name}')
        

def scheduler_factory(scheduler_name):
    if scheduler_name == 'MultiStepLR':
        return partial(
            torch.optim.lr_scheduler.MultiStepLR,
            milestones=[15, 16, 17, 18, 19], 
            gamma=0.5)
    elif scheduler_name == 'CyclicLR':
        return partial(
            torch.optim.lr_scheduler.CyclicLR, 
            base_lr=0.001, max_lr=0.1,
            step_size_up=3, mode="exp_range",
            gamma=0.85)
    elif scheduler_name == 'OneCycleLR':
        return partial(
            torch.optim.lr_scheduler.OneCycleLR, max_lr=0.1, steps_per_epoch=10, epochs=10)
    else:
        return None
        
def calculate_accuracy(pred, y):
    pred = torch.argmax(pred, dim=1)
    acc = (pred == y).sum() / len(y)
    return acc


class Classifier(pl.LightningModule):
    def __init__(
        self,model_name, 
        loss_name, loss_args, optimizer_name,
        scheduler_name, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_factory(model_name)
        self.lr = lr
        self.loss_fn = loss_factory(loss_name, *loss_args)
        self.optimizer = optimizer_factory(optimizer_name) 
        self.scheduler = scheduler_factory(scheduler_name)
        self.acc = calculate_accuracy
        
    def forward(self, x):
        x = self.model(x)
        x = torch.Tensor(x)
        return x
        
    def training_step(self, batch, batch_idx):
        # print(self.lr_schedulers().get_last_lr())
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss_fn(
            predictions, targets)
        acc = self.acc(predictions, targets)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss_fn(predictions, targets)
        acc = self.acc(predictions, targets)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.scheduler:
            scheduler = {
                'scheduler': self.scheduler(optimizer)
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {'optimizer': optimizer}
    