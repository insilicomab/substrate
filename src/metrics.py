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
