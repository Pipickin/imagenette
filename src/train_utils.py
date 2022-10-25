import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# mpl.use('tkagg')


def get_columns_to_plot(df):
    val_columns = [column.split('_')[1:] for 
           column in df.columns if 'val' in column]
    train_columns = [column.split('_')[1:] for 
             column in df.columns if 'train' in column]
    columns_to_plot = []
    for val_column in val_columns:
        if val_column in train_columns:
            columns_to_plot.append(('val_' + '_'.join(val_column), 
                                'train_' + '_'.join(val_column)))
        else:
            columns_to_plot.append(('val_' + '_'.join(val_column), ))
    for train_column in train_columns:
        if train_column not in val_columns:
            columns_to_plot.append(('train_' + '_'.join(train_column),))
    return columns_to_plot


def plot_metrics(metrics_df_path, save_path=''):
    print(metrics_df_path)
    df = pd.read_csv(metrics_df_path)
    columns_to_plot = get_columns_to_plot(df)
    for colums in columns_to_plot:
        for column in colums:
            column_df = df[df[column].notna()]
            plot = sns.lineplot(data=column_df, x='step', y=column)
        plt.legend(labels=[*colums])
        if save_path:
            plot.get_figure().savefig(
                os.path.join(save_path, '_'.join(column.split('_')[1:]) + '.png'))
        # plt.show()
        plt.clf()
        
def get_logger(model_name, optimizer_name, loss_name, version):
    csv_logger = CSVLogger(
        "logs", 
        name=model_name,
        version=f"{optimizer_name}_{loss_name}_{version}")
    return csv_logger


        
def get_optuna_logger(
    model_name,     
    loss_name,
    loss_args,
    optimizer_name,
    scheduler_name,
    lr
):
    csv_logger = CSVLogger(
        "logs", 
        name=model_name,
        version=f"{loss_name}_{loss_args}_{optimizer_name}_{scheduler_name}_{lr}")
    return csv_logger


def get_checkpoint_callback(model_name, optimizer_name, loss_name, version):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'models/{model_name}/{optimizer_name}_{loss_name}_{version}',
        filename=f'model',
        # save_weights_only=True,
        save_weights_only=False,
        mode="min")
    return checkpoint_callback


def get_logger_and_checkpoint_callback(
    model_name, optimizer_name,
    loss_name, version):
    logger = get_logger(
        model_name, optimizer_name,
        loss_name, version)
    checkpoint_callback = get_checkpoint_callback(
        model_name, optimizer_name,
        loss_name, version)
    return logger, checkpoint_callback
    

def get_extra_data(metrics_df_path):
    extra_data = {}
    df = pd.read_csv(metrics_df_path)
    columns = [column for column in df.columns if 
               'val' in column or 'train' in column]
    for column in columns:
        column_min = df[column].min()
        column_max = df[column].max()
        extra_data[column] = {
            'min': column_min,
            'max': column_max
        }
    return extra_data

