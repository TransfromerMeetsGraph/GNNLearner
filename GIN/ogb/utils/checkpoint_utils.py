from pathlib import Path

import torch


def restore_checkpoint(model: 'torch.nn.Module', optimizer: 'torch.optim.Optimizer', lr_scheduler,
                       checkpoint_dir: str,
                       checkpoint_name: str, reset_optimizer: bool = False, reset_lr_scheduler: bool = False):
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    if not checkpoint_dir or not checkpoint_path.exists():
        print('| No exist checkpoints, train from scratch')
        return 0
    checkpoint = torch.load(checkpoint_path)

    print(f'| Restore from {checkpoint_path} (epoch {checkpoint["epoch"]})')
    model.load_state_dict(checkpoint['model_state_dict'])

    if not reset_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if not reset_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    return start_epoch
