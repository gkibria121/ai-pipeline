
# ============================================================================
# FILE: utils/optimizer.py
# ============================================================================

import random
import numpy as np
import torch
import torch.nn as nn


def str_to_bool(val):
    """Convert string to boolean"""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError(f'Invalid truth value: {val}')


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine annealing learning rate decay"""
    return lr_min + (lr_max - lr_min) * 0.5 * \
           (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Keras-style learning rate decay"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGDR (Stochastic Gradient Descent with Warm Restarts) scheduler"""
    
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Create optimizer from config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=optim_config['base_lr'],
            momentum=optim_config.get('momentum', 0.9),
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config.get('nesterov', True)
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optim_config['base_lr'],
            betas=optim_config.get('betas', [0.9, 0.999]),
            weight_decay=optim_config['weight_decay'],
            amsgrad=str_to_bool(str(optim_config.get('amsgrad', 'False')))
        )
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """Create learning rate scheduler from config"""
    scheduler_name = optim_config.get('scheduler', None)
    
    if scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay']
        )
    elif scheduler_name == 'sgdr':
        scheduler = SGDRScheduler(
            optimizer,
            optim_config['T0'],
            optim_config['Tmult'],
            optim_config['lr_min']
        )
    elif scheduler_name == 'cosine':
        total_steps = optim_config['epochs'] * \
                     optim_config['steps_per_epoch']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step, total_steps, 1,
                optim_config['lr_min'] / optim_config['base_lr']
            )
        )
    elif scheduler_name == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: keras_decay(step)
        )
    else:
        scheduler = None
    
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Create optimizer and scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """Set seed for dataloader worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config):
    """Set random seed for reproducibility"""
    if config is None:
        raise ValueError("config cannot be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(
            config.get("cudnn_deterministic_toggle", "True")
        )
        torch.backends.cudnn.benchmark = str_to_bool(
            config.get("cudnn_benchmark_toggle", "False")
        )