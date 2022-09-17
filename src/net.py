import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

import timm

from omegaconf import DictConfig

from .optimizer import get_optimizer
from .losses import get_loss_fn


class Net(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # model
        self.model = timm.create_model(
            self.cfg.net.model_name,
            pretrained=self.cfg.net.pretrained,
            num_classes=self.cfg.num_classes
        )
        
        # loss function
        self.loss_fn = get_loss_fn(cfg=self.cfg)

        # metrics
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.Precision(
                    num_classes=self.cfg.num_classes,
                    average=self.cfg.metrics.precision.average,
                ),
                torchmetrics.Recall(
                    num_classes=self.cfg.num_classes,
                    average=self.cfg.metrics.recall.average,
                ),
                torchmetrics.Specificity(
                    num_classes=self.cfg.num_classes,
                    average=self.cfg.metrics.specificity.average,
                ),
                torchmetrics.F1Score(
                    num_classes=self.cfg.num_classes,
                    average=self.cfg.metrics.f1.average,
                ),
                torchmetrics.FBetaScore(
                    num_classes=self.cfg.num_classes,
                    beta=self.cfg.metrics.f_beta.beta,
                    average=self.cfg.metrics.f_beta.average,
                ),
                torchmetrics.AUROC(
                    num_classes=self.cfg.num_classes,
                    average=self.cfg.metrics.auroc.average,
                )
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

        
    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        preds = F.softmax(y, dim=1)

        loss = self.loss_fn(y, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_metrics(preds, t)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        preds = F.softmax(y, dim=1)

        loss = self.loss_fn(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_metrics(preds, t)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    
    def configure_optimizers(self):
        optimizer = get_optimizer(cfg=self.cfg, net=self.model)
        return optimizer


