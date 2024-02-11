import torch
import argparse
import os
from dataloader import DataLoader
import numpy as np
import time
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
import pandas as pd
import json


def sparse2torch_sparse(data): # encoder F.normalize에서 대신 활용
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0] # n_users
    features = data.shape[1] # n_items
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1)) # 유저별로 interaction 수만큼 normalize
    # row_norms_inv = 1 / np.linalg.norm(data, axis=1) # 유저별로 interaction 수만큼 normalize
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return torch.FloatTensor(t.to_dense())

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train(args, model, criterion, optimizer, train_data, epoch, update_count, is_VAE = False):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    N = train_data.shape[0]
    idxlist = list(range(N))

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]] # 여기의 train_data: sparse interaction matrix 형태
        data = naive_sparse2tensor(data).to(args.device)
        # data = sparse2torch_sparse(data).to(device)
        optimizer.zero_grad()

        if is_VAE:
          if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                            1. * update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)

          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))


            start_time = time.time()
            train_loss = 0.0
    
    return update_count


def evaluate(args, model, criterion, data_tr, data_te, update_count, is_VAE=False):
    # Turn on evaluation mode
    model.eval()
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    total_val_loss_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, e_N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
            # data_tensor = sparse2torch_sparse(data).to(device)
            if is_VAE :

              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap,
                                1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap

              recon_batch, mu, logvar = model(data_tensor)

              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)

            total_val_loss_list.append(loss.item())

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)
            # top10 = torch.topk(recon_batch)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return np.nanmean(total_val_loss_list), np.nanmean(n100_list), np.nanmean(r20_list), np.nanmean(r50_list)


def test(args, model, criterion, test_data_tr, test_data_te, update_count):
    # Run on test data.
    test_loss, n100, r20, r50 = evaluate(args, model, criterion, test_data_tr, test_data_te, update_count, is_VAE=True)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r20, r50))
    print('=' * 89)


def inference(args, model, data, current_time, is_VAE=False):
   
    # id
    with open('json_id/id2show.json', 'r') as json_file:
        id2show = json.load(json_file)
    id2show = json.loads(id2show)

    with open('json_id/id2profile.json', 'r') as json_file:
        id2profile = json.load(json_file)
    id2profile = json.loads(id2profile)


    # turn on eval mode
    model.eval()
    idxlist = list(range(data.shape[0]))
    N = data.shape[0]
    total_topk = []
    result = pd.DataFrame()

    with torch.no_grad():
        for start_idx in range(0, N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            batch = data[idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
            # data_tensor = sparse2torch_sparse(batch).to(args.device)
            if is_VAE :
              recon_batch, mu, logvar = model(data_tensor)
            else :
              recon_batch = model(data_tensor)

            # Exclude examples from training set
            recon_batch = recon_batch
            recon_batch[batch.nonzero()] = -1e9

            # top-k choice
            topk = torch.topk(input=recon_batch, k=10)
            batch_topk = list(topk.indices.reshape(-1).detach().cpu().numpy())
          
            total_topk.extend(batch_topk)


    result['user'] = np.repeat(np.arange(31360),10)
    result['item'] = total_topk

    result['user'] = result['user'].apply(lambda x: id2profile[x])
    result['item'] = result['item'].apply(lambda x: id2show[x])
    
    result_df = result.sort_values('user').reset_index(drop=True)
    result_df.to_csv(f'submission/{args.model}_{current_time}.csv', index=False)


def verbose(epoch, epoch_start_time, val_loss, n100, r20, r50):
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n100, r20, r50))
        print('-' * 89)