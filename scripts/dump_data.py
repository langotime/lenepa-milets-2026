import argparse
from os import path

import numpy as np

from data.datasets import DATASETS, PTB_XL
from data.utils import load_raw_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to data directory')
parser.add_argument('--dataset', choices=list(DATASETS), default='ptb-xl', help='dataset type')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()

args.data_dir = path.normpath(args.data_dir)
if args.dataset != 'ptb-xl':
  raise ValueError('This reproduction repo only supports the PTB-XL dump path.')

print(f'Loading data from {args.data_dir}')

record_names = PTB_XL.find_records(args.data_dir)
data = load_raw_data(record_names, verbose=args.verbose)

out_file = f'{args.data_dir}.npy'
print(f'Saving dataset to {out_file}')
np.save(out_file, data)
