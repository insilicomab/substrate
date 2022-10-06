import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torchmetrics
from omegaconf import DictConfig


def get_metrics(cfg: DictConfig):
    metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.Precision(
                    num_classes=cfg.num_classes,
                    average=cfg.metrics.precision.average,
                ),
                torchmetrics.Recall(
                    num_classes=cfg.num_classes,
                    average=cfg.metrics.recall.average,
                ),
                torchmetrics.Specificity(
                    num_classes=cfg.num_classes,
                    average=cfg.metrics.specificity.average,
                ),
                torchmetrics.F1Score(
                    num_classes=cfg.num_classes,
                    average=cfg.metrics.f1.average,
                ),
                torchmetrics.FBetaScore(
                    num_classes=cfg.num_classes,
                    beta=cfg.metrics.f_beta.beta,
                    average=cfg.metrics.f_beta.average,
                ),
            ]
        )

    return metrics


def get_classification_metrics(y_true, y_pred_proba, y_pred, t_onehot, cfg: DictConfig):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=cfg.metrics.precision.average)
    recall = recall_score(y_true, y_pred, average=cfg.metrics.recall.average)
    f1 = f1_score(y_true, y_pred, average=cfg.metrics.f1.average)
    specificity = torchmetrics.functional.specificity(
        torch.tensor(y_true),
        torch.tensor(y_pred),
        num_classes=cfg.num_classes,
        average=cfg.metrics.specificity.average
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = roc_auc_score(t_onehot, y_pred_proba, average="macro")

    results = (accuracy, precision, recall, f1, specificity, kappa, auc)

    return results


def get_classification_report(y_true, y_pred):
    cls_report = classification_report(y_true, y_pred)
    return cls_report


def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm