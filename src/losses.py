from torch import nn
from omegaconf import DictConfig


def get_loss_fn(cfg: DictConfig):
    if cfg.loss_fn.name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f'Unknown optimizer: {cfg.loss_fn.name}')