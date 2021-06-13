#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Some other LR schedulers."""

from torch.optim.lr_scheduler import LambdaLR, StepLR


def build_lr_scheduler(lr_scheduler_string: str, optimizer, train_subset: bool):
    """Build LR scheduler.


    (default|step)[:s=30,g=0.25]
    inverse_sqrt[:w=4000,init=1e-3]
    """
    def _get_lr_s_args():
        if ':' in lr_scheduler_string:
            _, lr_s_args_str = lr_scheduler_string.split(':')
            lr_s_args = lr_s_args_str.split(',')
        else:
            lr_s_args = []
        return lr_s_args

    if lr_scheduler_string.startswith(('default', 'step')):
        lr_s_args = _get_lr_s_args()
        step_size = 30
        gamma = 0.25

        for lr_s_arg in lr_s_args:
            k, v = lr_s_arg.split('=')
            if k == 's':
                step_size = int(v)
            elif k == 'g':
                gamma = float(v)
        if train_subset:
            step_size *= 10

        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_string.startswith('inverse_sqrt'):
        # [NOTE]: --train_subset option has no effect on inverse_sqrt scheduler.
        lr_s_args = _get_lr_s_args()
        w = 4000
        init = -1
        for lr_s_arg in lr_s_args:
            k, v = lr_s_arg.split('=')
            if k == 'w':
                w = int(v)
            elif k == 'init':
                init = float(v)
        scheduler = create_inverse_square_root_lr_scheduler(
            optimizer, warmup_epochs=w, warmup_init_factor=init)
    else:
        raise ValueError(f'Unknown LR scheduler string {lr_scheduler_string}')
    return scheduler


def create_inverse_square_root_lr_scheduler(optimizer, warmup_epochs=4000, warmup_init_factor=-1):
    if warmup_init_factor < 0:
        warmup_init_factor = 0 if warmup_epochs > 0 else 1.0

    if warmup_epochs > 0:
        lr_factor_step = (1.0 - warmup_init_factor) / warmup_epochs
    else:
        lr_factor_step = 0
    decay_factor = warmup_epochs ** 0.5

    # [NOTE]: The LR lambda function returns a learning rate **FACTOR**, not learning rate itself.
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_init_factor + epoch * lr_factor_step
        else:
            return decay_factor * (epoch ** -0.5)

    return LambdaLR(optimizer, _lr_lambda)
