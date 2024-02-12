import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from utils import argparsing
import pandas as pd
from dataloader import DataLoader
from models import MultiDAE, MultiVAE, loss_function_dae, loss_function_vae
from trainers import vae_train, vae_evaluate, recvae_train, recvae_evaluate, test, inference, verbose
import time


def multi_vae_runner(args, model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, epoch, N):
    epoch_start_time = time.time()

    vae_train(args, model, criterion, optimizer, train_data, epoch)
    val_loss, n100, r10, r20, r50 = vae_evaluate(args, model, criterion, vad_data_tr, vad_data_te)
    verbose(epoch, epoch_start_time, val_loss, n100, r10, r20, r50)
    
    n_iter = epoch * len(range(0, N, args.batch_size)) # 이게 뭐지?

    return n100


def recvae_runner(args, model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, epoch, N):
    
    epoch_start_time = time.time()

    encoder_params = set(model.encoder.parameters())
    decoder_params = set(model.decoder.parameters())

    optimizer_encoder = optim.Adam(encoder_params, lr=args.lr)
    optimizer_decoder = optim.Adam(decoder_params, lr=args.lr)
    
    ## training
    for _ in range(args.encoder_epochs): # enc:dec = 3:1
        recvae_train(args, model, optimizer_encoder, train_data, epoch, 0.5)
    model.update_prior()
    recvae_train(args, model, optimizer_decoder, train_data, epoch, 0)
    val_loss, n100, r10, r20, r50 = recvae_evaluate(args, model, vad_data_tr, vad_data_te)
    verbose(epoch, epoch_start_time, val_loss, n100, r10, r20, r50)

    return n100

    