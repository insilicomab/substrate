import pandas as pd

import os

from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl
from pytorch_lightning import callbacks

import hydra
from omegaconf import DictConfig

from src import SubstrateDataModule, Net


@hydra.main(version_base=None, config_path='config', config_name='config_kfold')
def main(cfg: DictConfig):

    # read data frame
    train_master = pd.read_csv(cfg.train_master_dir)

    train_num_classes = train_master['flag'].nunique()
    assert (
        cfg.num_classes == train_num_classes
    ), f'num_classes should be {train_num_classes}'

    # image name list
    X_train = train_master['file_name']

    # label list
    Y_train = train_master['flag']

    # set seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    # directory to save models
    SAVE_MODEL_PATH = cfg.save_model_dir
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # K-fold
    skf = StratifiedKFold(
        n_splits=cfg.skfold.n_splits,
        shuffle=cfg.skfold.shuffle,
        random_state=cfg.skfold.random_state
    )

    for i, (train_index, val_index) in enumerate(skf.split(X_train, Y_train)):
        x_train = X_train.iloc[train_index].values
        x_val = X_train.iloc[val_index].values
        y_train = Y_train.iloc[train_index].values
        y_val = Y_train.iloc[val_index].values

    
        # datamodule
        datamodule = SubstrateDataModule(
            cfg=cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        net = Net(cfg=cfg)

        # Callbacks
        callback_list = []
        if cfg.callbacks.early_stopping.enable:
            earlystopping = callbacks.EarlyStopping(
                monitor=cfg.callbacks.early_stopping.monitor,
                patience=cfg.callbacks.early_stopping.patience,
                mode=cfg.callbacks.early_stopping.mode,
                verbose=True,
                strict=True,
            )
            callback_list.append(earlystopping)
        if cfg.callbacks.model_checkpoint.enable:
            model_checkpoint = callbacks.ModelCheckpoint(
                dirpath=SAVE_MODEL_PATH,
                filename=f'{cfg.net.model_name}-{i}',
                monitor=cfg.callbacks.model_checkpoint.monitor,
                mode=cfg.callbacks.model_checkpoint.mode,
                save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
                save_last=cfg.callbacks.model_checkpoint.save_last,
            )
            callback_list.append(model_checkpoint)
    
        # trainer
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            callbacks=callback_list,
            gpus=cfg.trainer.gpus,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            auto_lr_find=cfg.trainer.auto_lr_find
        )

        # train
        trainer.fit(net, datamodule=datamodule)

        # display current fold
        print(f'Current fold is {i}')


if __name__ == '__main__':
    main()