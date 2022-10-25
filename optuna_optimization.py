import multiprocessing as mp
import time 
from functools import partial

import torch
import cv2
import optuna
from omegaconf import DictConfig
from optuna.trial import Trial
import hydra

from train_script import train

    
def objective(trial: Trial, optimization_cfg):
    # choose variables
    # STANDARD OPTIMIZATION
    model_name = trial.suggest_categorical(
        "model_name", ["resnet50", "resnet18", "xresnet50"])
    loss_name = trial.suggest_categorical(
        "loss_name", ["cross_entropy", "smooth_cross_entropy"])
    if loss_name == "smooth_cross_entropy":
        loss_args = [trial.suggest_float(
            'loss_args', 0.1, 0.8)]
    else:
        loss_args = [.0]
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["radam", "adam", 'SGD'])
    if optimizer_name == 'SGD':
        scheduler_name = trial.suggest_categorical(
            "scheduler_name", ["MultiStepLR", "CyclicLR"])
    else:
        scheduler_name = "MultiStepLR"
    lr = trial.suggest_float('lr', 1e-4, 1e-3)

    score = train(
        model_name,
        loss_name,
        loss_args,
        optimizer_name,
        scheduler_name,
        lr,
        optimization_cfg)
    return score
    

    
@hydra.main(config_path='conf', config_name='config_optuna')
def main(cfg: DictConfig):
    study = optuna.create_study(
        direction='maximize',
        study_name=cfg.study_name,
        load_if_exists=True,
        storage=f'sqlite:///{cfg.study_name}.db')
    objective_with_cfg = partial(
        objective, optimization_cfg=cfg
    )
    study.optimize(
        objective_with_cfg, n_trials=cfg.total_trials)
    
    
if __name__ == '__main__':
    main()