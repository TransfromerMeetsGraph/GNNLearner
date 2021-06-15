#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Initialize the Multi-branch Roberta checkpoints with corresponding single-branch Roberta checkpoints.
Used for warm start MB Roberta training.
"""

import argparse
import collections
import logging
import re
import traceback

import torch

_ATTN_BRANCHES_KEY_FULL = re.compile(r'.*attn_branches\.\d+\..*')
_ATTN_BRANCHES_KEY = re.compile(r'attn_branches\.\d+\.')
_PFFN_BRANCHES_KEY_FULL = re.compile(r'.*fc[12]_branches\.\d+\..*')
_PFFN_BRANCHES_KEY = re.compile(r'fc([12])_branches\.\d+\.')

_STD_ATTN_BRANCHES_KEY_FULL = re.compile(r'.*attn\..*')
_STD_ATTN_BRANCHES_KEY = re.compile(r'attn\.')
_STD_PFFN_BRANCHES_KEY_FULL = re.compile(r'.*fc[12]\..*')
_STD_PFFN_BRANCHES_KEY = re.compile(r'fc([12])\.')


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser('Initialize the MB Roberta checkpoint with corresponding single-branch Transformer checkpoints.')
    parser.add_argument('path', help='Original checkpoint path.')
    parser.add_argument('output_path', help='The output checkpoint path.')
    parser.add_argument('-N', type=int, help='New N value')
    parser.add_argument('--NF', type=int, help='New NF value')
    parser.add_argument('--ro', '--reset-optimizer', dest='reset_optimizer',
                        action='store_true', default=False, help='Reset optimizer states.')
    parser.add_argument('--std-transformer', action='store_true', default=False,
                        help='The input checkpoint is a standard Transformer.')

    args = parser.parse_args()

    checkpoint = torch.load(args.path, map_location='cpu')
    model = checkpoint['model']
    pt_args = checkpoint['args']

    print('| Init MBT checkpoint: convert {oldN}-{oldNF}-{dH}-{dF} to {N}-{NF}-{dH}-{dF}'.format(
        oldN=getattr(pt_args, 'encoder_branches', 1), oldNF=getattr(pt_args, 'encoder_pffn_branches', 1),
        dH=pt_args.encoder_embed_dim, dF=pt_args.encoder_ffn_embed_dim,
        N=args.N, NF=args.NF))
    if args.std_transformer:
        print('| Input checkpoint is a standard Transformer.')
        _attn_branches_key_full = _STD_ATTN_BRANCHES_KEY_FULL
        _attn_branches_key_sub = _STD_ATTN_BRANCHES_KEY
        _pffn_branches_key_full = _STD_PFFN_BRANCHES_KEY_FULL
        _pffn_branches_key_sub = _STD_PFFN_BRANCHES_KEY
    else:
        _attn_branches_key_full = _ATTN_BRANCHES_KEY_FULL
        _attn_branches_key_sub = _ATTN_BRANCHES_KEY
        _pffn_branches_key_full = _PFFN_BRANCHES_KEY_FULL
        _pffn_branches_key_sub = _PFFN_BRANCHES_KEY
    
    new_model = collections.OrderedDict()
    
    for key, param in model.items():
        if _attn_branches_key_full.fullmatch(key) is not None:
            for i in range(args.N):
                new_key = _attn_branches_key_sub.sub(r'attn_branches.{}.'.format(i), key, count=1)
                new_model[new_key] = param.clone()
                print(f'{key} => {new_key}')
        elif _pffn_branches_key_full.fullmatch(key) is not None:
            for i in range(args.NF):
                new_key = _pffn_branches_key_sub.sub(r'fc\1_branches.{}.'.format(i), key, count=1)
                new_model[new_key] = param.clone()
                print(f'{key} => {new_key}')
        else:
            new_model[key] = param
    
    checkpoint['model'] = new_model
    pt_args.encoder_branches = args.N
    pt_args.encoder_pffn_branches = args.NF

    if args.reset_optimizer:
        checkpoint['extra_state'] = None
        checkpoint.pop('optimizer_history', None)
        checkpoint['last_optimizer_state'] = None
        checkpoint['warm_init'] = True

    torch_persistent_save(checkpoint, args.output_path)
    print('| Init MBT checkpoint: saved to {!r}.'.format(args.output_path))


if __name__ == "__main__":
    main()
