import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from run import argparsing
import pandas as pd
from modules_for_preprocess import filter_triplets, split_data, numerize_write

args = argparsing()

## read raw data
DATA_DIR = args.data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

## filtering
raw_data, user_activity, item_popularity, unique_uid = filter_triplets(raw_data, min_uc=5, min_sc=0)

## train/val/test split
train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te, show2id, profile2id = split_data(args, unique_uid, raw_data)

## numerize and write
numerize_write(args, profile2id, show2id, raw_data, train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te)