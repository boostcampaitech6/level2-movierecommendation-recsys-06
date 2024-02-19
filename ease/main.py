import os
import yaml
import wandb
import pandas as pd
import numpy as np
from trainer import train_valid_split,create_matrix_and_mappings, recall_at_10
from utils import korea_date_time, set_seeds
from model import EaseModel

seed = 42
set_seeds(seed)

# YAML 파일 로드
config_path = "config/ease.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# W&B sweep 설정
#sweep_id = wandb.sweep(sweep=config, project="MovieRec")

project_name = "MovieRec"
user_name = config["user_name"]
model_name = "EASE"
date_time = korea_date_time()

def main():
    # wandb
    wandb.login(key=config["wandb_key"])


    wandb_args = {
            "_lambda": config["_lambda"],
            "scale": config["scale"],
        }

    wandb.init(
            project=project_name,
            name=f"{user_name}-{model_name}-{date_time}",
            config=wandb_args,
        )
    
    wandb.run.name = f"{date_time} yechan"
    
    print("###Load data###")
    train_df = pd.read_csv(os.path.join(config["data_path"], "train_ratings.csv"))

    print("###Split data###")
    train_data, valid_data = train_valid_split(train_df, num_seq=4, num_ran=6)

    print("###Evaluating with valid data###")
    X_cv, index_to_user, index_to_item = create_matrix_and_mappings(
            train_data, config["scale"]
        )
    model_cv = EaseModel(_lambda=config["_lambda"])
    model_cv.train(X_cv)
    result = model_cv.forward(X_cv[:, :])
    result[X_cv.nonzero()] = -np.inf

    # Extract top 10
    recommend_list = []
    for i in range(len(result)):
        sorted_indices = np.argsort(-result[i])
        for j in sorted_indices.tolist():
            for k in range(10):
                recommend_list.append((index_to_user[i], index_to_item[j[k]]))

    pred_df = pd.DataFrame(recommend_list, columns=["user", "item"])

    # Evaluate Recall@10 performance
    recall_10 = recall_at_10(true_df=valid_data, pred_df=pred_df)
    print("Recall@10:", recall_10)
    wandb.log({"Recall@10": recall_10})

    # train with total data
    print("###Preprocessing###")
    X, _, _ = create_matrix_and_mappings(train_df, config["scale"])
    
    print("###Training###")
    model = EaseModel(_lambda=config["_lambda"])
    model.train(train_df)

    print('###Inference###')
    predict = model.predict(train_df, train_df['user'].unique(), train_df['item'].unique(), 10)
    predict = predict.drop('score', axis = 1)
    predict.to_csv(f'output/{date_time} EASE.csv', index=False)

    print("Submission file has been made")
    
main()

#wandb.agent(sweep_id, main)