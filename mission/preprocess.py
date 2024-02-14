import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from utils import argparsing
import pandas as pd
from modules_for_preprocess import filter_triplets, split_data, numerize_write, split_data2, numerize_write2

args = argparsing()

## read raw data
DATA_DIR = args.data
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)



### train(8) : val(1) : test(1)
args.pro_dir = os.path.join(args.data_dir, 'pro_sg')

## filtering
raw_data, user_activity, item_popularity, unique_uid = filter_triplets(raw_data, min_uc=5, min_sc=0)

## train/val/test split
train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te, show2id, profile2id = split_data(args, unique_uid, raw_data)

## numerize and write
numerize_write(args, profile2id, show2id, raw_data, train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te)



### train(9) : test(1)
args.pro_dir = os.path.join(args.data_dir, 'pro_sg2')

## filtering
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

## train/val split
train_plays, vad_plays_tr, vad_plays_te, show2id, profile2id = split_data2(args, unique_uid, raw_data)

## numerize and write
numerize_write2(args, profile2id, show2id, raw_data, train_plays, vad_plays_tr, vad_plays_te)