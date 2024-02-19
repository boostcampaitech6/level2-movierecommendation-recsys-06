import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed, korea_date_time,
    generate_submission_file
)

import wandb
import yaml


def train():
    parser = argparse.ArgumentParser()

    default_config={
    "hidden_size": 256,
    "attention_probs_dropout_prob": 0.5,
    "hidden_dropout_prob": 0.5,
    "max_seq_length": 300,
    "lr": 0.001,
    "batch_size": 256,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    }    

    wandb.init(config=default_config)

    parser.add_argument("--data_dir", default="../data/ntrain/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=wandb.config.hidden_size, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=wandb.config.attention_probs_dropout_prob,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=wandb.config.hidden_dropout_prob, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=wandb.config.max_seq_length, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=wandb.config.lr, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=wandb.config.batch_size, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=wandb.config.adam_beta1, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=wandb.config.adam_beta2, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument(
        "--using_pretrain", type=bool, help="use to pretrain"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    
    args.date_time = korea_date_time()
    wandb.run.name = f"{args.date_time} yechan"
        
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"

    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    print(args.using_pretrain)
    if args.using_pretrain:
        pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=7, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    #print(result_info)


    #inference
    print("---------------Inference-------------------")
    submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)

    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    preds= trainer.submission(0)

    generate_submission_file(args, preds)

# sweep config
sweep_config_path = 'sweepconfig.yaml'
with open(sweep_config_path, 'r') as file:
    sweep_config = yaml.safe_load(file)

wandb.login()
sweep_id = wandb.sweep(sweep=sweep_config, project="MovieRec", entity="boostcamp6-recsys6",)

wandb.agent(sweep_id, train)

