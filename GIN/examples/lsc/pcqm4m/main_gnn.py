import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

### importing OGB-LSC
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
from ogb.utils.checkpoint_utils import restore_checkpoint
from ogb.utils.lr_schedulers import build_lr_scheduler

reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, args):
    model.train()
    loss_accum = 0

    itr = loader
    if not args.no_tqdm:
        itr = tqdm(loader, desc="Iteration")

    for step, batch in enumerate(itr):
        batch = batch.to(device)

        pred = model(batch).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, args):
    model.eval()
    y_true = []
    y_pred = []

    itr = loader
    if not args.no_tqdm:
        itr = tqdm(loader, desc="Iteration")

    for step, batch in enumerate(itr):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader, args):
    model.eval()
    y_pred = []

    itr = loader
    if not args.no_tqdm:
        itr = tqdm(loader, desc="Iteration")

    for step, batch in enumerate(itr):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--residual', nargs='?', type=bool, const=False, default=False,
                        help='residual connection (default: False)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=str, default='100',
                        help='number of epochs to train, with optional prefix "+" (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='directory to dataset')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='dataset name (used for scaffold and other new datasets) (default: None)')
    parser.add_argument('--no-tqdm', action='store_true', help='disable tqdm progress bar')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr-scheduler', type=str, default='default',
                        help='LR scheduler string (default: "default")')
    parser.add_argument('--dev-kfold-tune', nargs=2, type=int, default=None, help='Dev tuning (fold-id total-folds)')
    parser.add_argument('--reset-optimizer', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--reset-lr-scheduler', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygPCQM4MDataset(root=args.dataset_dir, dataset_name=args.dataset_name)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    if getattr(args, 'dev_kfold_tune', None) is not None:
        valid_fold_id, num_folds = args.dev_kfold_tune
        extra_name = f'_kfold_{valid_fold_id}_{num_folds}_lr{args.lr}_ro{args.reset_optimizer}_rs{args.reset_lr_scheduler}'
    else:
        valid_fold_id, num_folds = None, None
        extra_name = ''

    if valid_fold_id is not None:
        assert not args.train_subset, '--dev-tune does not support --train-subset'
        valid_size = split_idx['valid'].numel()
        fold_size = valid_size // num_folds + (valid_size % num_folds != 0)
        folds = torch.split(split_idx['valid'], fold_size)
        valid_fold = folds[valid_fold_id]

        train_subset = torch.cat([split_idx['train']] + [folds[i] for i in range(len(folds)) if i != valid_fold_id])
        train_loader = DataLoader(dataset[train_subset], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[valid_fold], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    else:
        if args.train_subset:
            subset_ratio = 0.1
            subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]
            train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        else:
            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)

    if args.save_test_dir is not '':
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    if args.checkpoint_dir is not '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'residual': args.residual,
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Setup LR scheduler.
    scheduler = build_lr_scheduler(args.lr_scheduler, optimizer, train_subset=args.train_subset)
    start_epoch = restore_checkpoint(
        model, optimizer, scheduler, args.checkpoint_dir, 'checkpoint.pt',
        reset_optimizer=args.reset_optimizer, reset_lr_scheduler=args.reset_lr_scheduler,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    if args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    # Setup #epochs
    epoch_value = int(args.epochs.lstrip('+'))
    if args.epochs.startswith('+'):
        num_epochs = start_epoch + epoch_value
    else:
        num_epochs = epoch_value
    if args.train_subset:
        num_epochs = 1000

    for epoch in range(start_epoch + 1, num_epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer, args)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator, args)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir:
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir:
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}
                if valid_fold_id is not None:
                    ckpt_name = f'checkpoint{extra_name}.pt'
                else:
                    ckpt_name = 'checkpoint.pt'
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, ckpt_name))

            if args.save_test_dir:
                print('Predicting on test data...')
                y_pred = test(model, device, test_loader, args)
                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_test_dir, extra_name=extra_name)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')
        print(flush=True)

    if args.log_dir:
        writer.close()


if __name__ == "__main__":
    main()
