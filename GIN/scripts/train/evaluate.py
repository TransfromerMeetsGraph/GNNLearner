#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Get ensemble predictions."""

import argparse
import csv
import glob
import gzip
import pickle

from pathlib import Path

import numpy as np

TRAIN_SIZE = 3045360
VALID_SIZE = 380670
TEST_SIZE = 377423


def _load_orig_data(dataset_root_dir: Path):
    with gzip.open(dataset_root_dir / 'data.csv.gz', 'rt', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    train_rows = rows[:TRAIN_SIZE]
    valid_rows = rows[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]
    test_rows = rows[TRAIN_SIZE + VALID_SIZE:]

    data = {
        'train': {
            'idx': np.asarray([int(row[0]) for row in train_rows]),
            'smiles': [row[1] for row in train_rows],
            'homo-lumo': np.asarray([float(row[2]) for row in train_rows]),
        },
        'valid': {
            'idx': np.asarray([int(row[0]) for row in valid_rows]),
            'smiles': [row[1] for row in valid_rows],
            'homo-lumo': np.asarray([float(row[2]) for row in valid_rows]),
        },
        'test': {
            'idx': np.asarray([int(row[0]) for row in test_rows]),
            'smiles': [row[1] for row in test_rows],
            'homo-lumo': None,
        },
    }
    return data


def _load_data(dataset_root_dir: Path, subset: str):
    processed_filename = dataset_root_dir / f'{subset}.pkl'
    if not processed_filename.exists():
        data = _load_orig_data(dataset_root_dir)
        subset_data = data[subset]
        with open(processed_filename, 'wb') as f:
            pickle.dump(subset_data, f)
    else:
        with open(processed_filename, 'rb') as f:
            subset_data = pickle.load(f)
    return subset_data


def _load_predictions(pred_patterns: list):
    y_pred_list = []
    pred_filenames = []
    for pred_pattern in pred_patterns:
        pred_file_list = glob.glob(pred_pattern)

        for pred_file in pred_file_list:
            pred_filenames.append(pred_file)
            if pred_file.endswith(('.npy', '.npz', '.pkl')):
                np_data = np.load(pred_file)
                y_pred = np_data.get('y_pred', None)
                if y_pred is None:
                    y_pred = np_data.get('arr_0')
            else:
                y_pred = np.loadtxt(pred_file)
            y_pred_list.append(y_pred)
    return y_pred_list, pred_filenames


def _avg_y_pred(y_pred_list):
    return np.mean(y_pred_list, axis=0)


def _eval(y_pred, y_true):
    return float(np.mean(np.absolute(y_pred - y_true)))


def _dump_output(y_pred_mean, output_filename):
    if output_filename is None:
        return
    np.savez_compressed(output_filename, y_pred=y_pred_mean)
    print(f'Ensemble prediction output to {output_filename}.')


def main():
    parser = argparse.ArgumentParser('Get ensemble & evaluation predictions.')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='dataset folder')
    parser.add_argument('-s', '--subset', type=str, choices=['train', 'valid', 'test'], default='test',
                        help='evaluation subset')
    parser.add_argument('-p', '--pred', nargs='+', required=True,
                        help='prediction filename patterns, allow wildcard glob matching')
    parser.add_argument('-o', '--output', default=None, help='output ensemble prediction filename')

    args = parser.parse_args()
    print(args)

    dataset_root_dir = Path(args.dataset_dir) / 'pcqm4m_kddcup2021' / 'raw'
    subset_data = _load_data(dataset_root_dir, args.subset)

    y_pred_list, pred_filenames = _load_predictions(args.pred)
    y_pred_mean = _avg_y_pred(y_pred_list)
    _dump_output(y_pred_mean, args.output)

    if args.subset != 'test':
        y_true = subset_data['homo-lumo']
        if len(y_pred_list) > 1:
            for i, y_pred in enumerate(y_pred_list, start=1):
                mae = _eval(y_pred, y_true)
                print(pred_filenames[i - 1])
                print(f'Single MAE-{i}: {mae}')
        mae = _eval(y_pred_mean, y_true)
        print(f'> Ensemble MAE:', mae)


if __name__ == '__main__':
    main()
