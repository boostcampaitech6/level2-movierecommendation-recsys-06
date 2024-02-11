import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from run import argparsing
import pandas as pd
from dataloader import DataLoader
from models import MultiDAE, MultiVAE, loss_function_dae, loss_function_vae
from trainer import train, evaluate, test, inference, verbose
import time


## setting
current_time = time.strftime('%y%m%d_%H%M%S')
args = argparsing()
torch.manual_seed(args.seed)


## Load data
loader = DataLoader(args.data)
n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, data_inf = loader.data_loading()
N = train_data.shape[0]


## Build the model
p_dims = [200, 600, n_items]
models = {'MultiDAE': MultiDAE(p_dims), 'MultiVAE': MultiVAE(p_dims)}
losses = {'MultiDAE': loss_function_dae, 'MultiVAE':loss_function_vae}

model = models[args.model].to(args.device)
criterion = losses[args.model]
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


## Training
best_n100 = -np.inf
update_count = 0

for epoch in range(1, args.epochs):
    epoch_start_time = time.time()

    update_count = train(args, model, criterion, optimizer, train_data, epoch, update_count, is_VAE=True)
    val_loss, n100, r20, r50 = evaluate(args, model, criterion, vad_data_tr, vad_data_te, update_count, is_VAE=True)
    verbose(epoch, epoch_start_time, val_loss, n100, r20, r50)
    
    n_iter = epoch * len(range(0, N, args.batch_size))

    # Save the model if the n100 is the best we've seen so far.
    if n100 > best_n100:
        with open(f'model_files/{args.model} {current_time}.pt', 'wb') as f:
            torch.save(model, f)
        best_n100 = n100
        best_epoch = epoch
print('best epoch:', best_epoch)
print(f'best score n100:{best_n100}')

## load best model
with open(args.save, 'rb') as f:
    model = torch.load(f)

## Test
test(args, model, criterion, test_data_tr, test_data_te, update_count)

## Inference
inference(args, model, data_inf, current_time, is_VAE=False)
