import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from omegaconf import DictConfig

from .optimizer import get_optimizer
from .losses import get_loss_fn
from .metrics import (
    get_metrics, get_confusion_matrix,
    get_classification_metrics, get_classification_report,
)
from .scheduler import get_scheduler


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
        metrics = get_metrics(cfg=self.cfg)
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
    

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        y_pred_proba = F.softmax(y, dim=1)
        y_pred = torch.argmax(y, dim=1)
        t_onehot = torch.eye(self.cfg.num_classes)[t]
        outputs = {
            'true': t.to('cpu').squeeze(),
            'pred_proba': y_pred_proba.to('cpu'),
            'pred': y_pred.to('cpu').squeeze(), 
            'true_onehot': t_onehot.to('cpu'),
        }
        return outputs
    

    def test_epoch_end(self, outputs):
        y_true = np.array([output['true'] for output in outputs])

        y_pred_proba = [output['pred_proba'] for output in outputs]
        y_pred_proba = torch.cat(y_pred_proba, dim=0)

        y_pred = np.array([output['pred'] for output in outputs])

        t_onehot = [output['true_onehot'] for output in outputs]
        t_onehot = torch.cat(t_onehot, dim=0)

        self._evaluate((y_true, y_pred_proba, y_pred, t_onehot))

    
    def configure_optimizers(self):
        optimizer = get_optimizer(cfg=self.cfg, net=self.model)
        scheduler = get_scheduler(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]


    def _evaluate(self, ys):
        y_true, y_pred_proba, y_pred, t_onehot = ys
        self._log_metrics(y_true, y_pred_proba, y_pred, t_onehot)
        self._save_confusion_matrix(y_true, y_pred)
        self._save_classification_report(y_true, y_pred)
    

    def _log_metrics(self, y_true, y_pred_proba, y_pred, t_onehot):
        results = get_classification_metrics(y_true, y_pred_proba, y_pred, t_onehot, cfg=self.cfg)
        accuracy, precision, recall, f1, specificity, kappa, auc = results
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_fbeta', f1)
        self.log('test_specifity', specificity)
        self.log('test_kappa', kappa)
        self.log('test_auc', auc)
    

    def _save_classification_report(self, y_true, y_pred):
        """
        Save classification report to txt.
        """
        cls_report = get_classification_report(y_true, y_pred)
        with open('output/classification_report.txt', 'w') as f:
            f.write(cls_report)
        
        
    def _save_confusion_matrix(self, y_true, y_pred):
        """
        Save confusion matrix.
        """
        cm = get_confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Prediction label')
        plt.ylabel('True label')
        plt.savefig('output/confusion_matrix.png')