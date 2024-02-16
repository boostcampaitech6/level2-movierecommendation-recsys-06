import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, SASRec, BERT4Rec
from recbole.model.general_recommender import LightGCN, MultiDAE
from recbole.trainer import Trainer
import wandb
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_scores
import torch

# WandB 스윕 설정
sweep_config = {
    "method": "random",
    "metric": {
        "name": "recall@10",  # 이것은 실제로 대회에서 사용하는 메트릭으로 설정되어야 합니다.
        "goal": "maximize",
    },
    "parameters": {
        "model_name": {
            "values": ["GRU4Rec", "SASRec", "BERT4Rec", "LightGCN", "MultiDAE"]
        },
        "learning_rate": {"min": 0.001, "max": 0.01},
        "embedding_size": {"values": [32, 64, 128]},
        # 모델별로 다른 하이퍼파라미터를 추가.
    },
}

# 모델 선택 함수
def select_model(model_name, config, train_data):
    if model_name == "GRU4Rec":
        return GRU4Rec(config, train_data.dataset)
    elif model_name == "SASRec":
        return SASRec(config, train_data.dataset)
    elif model_name == "BERT4Rec":
        return BERT4Rec(config, train_data.dataset)
    elif model_name == "LightGCN":
        return LightGCN(config, train_data.dataset)
    elif model_name == "MultiDAE":
        return MultiDAE(config, train_data.dataset)
    else:
        raise ValueError("Unsupported model name!")


# WandB 스윕 훈련 함수
def sweep_train():
    global model_name, metric_value  # 전역 변수 사용
    with wandb.init() as run:
        config_dict = dict(run.config)
        config_dict["model"] = run.config["model_name"]
        model_name = config_dict["model"]  # 모델 이름 저장
        config = Config(
            model=config_dict["model"], dataset="./data/train", config_dict=config_dict
        )
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = select_model(run.config["model_name"], config, train_data)
        trainer = Trainer(config, model)
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)

        trainer.fit(train_data, valid_data, saved=True)

        test_result = trainer.evaluate(test_data)
        recall_at_10 = test_result["recall@10"]
        metric_value = recall_at_10  # 메트릭 값 저장
        wandb.log({"recall@10": recall_at_10})


# 데이터셋 로드
df = pd.read_csv("./data/train/train_ratings.csv")  # 실제 데이터셋 경로로 수정하세요.

# WandB 스윕 시작
sweep_id = wandb.sweep(
    sweep_config, project="level2-movie_rec_RecBole", entity="boostcamp6-recsys6"
)
trained_model, dataset = wandb.agent(sweep_id, sweep_train, count=1)  # 모델 훈련 및 객체 반환 받기

# 추천 생성 및 제출 파일 저장 함수
def generate_recommendations(model, dataset, user_list, top_k=10):
    final_recommendations = {}
    for user in user_list:
        _, user_idx = dataset.prepared_data["user2id"].get(user, (None, None))
        if user_idx is not None:
            user_idx_tensor = torch.LongTensor([user_idx]).to(dataset.config["device"])
            scores = full_sort_scores(model, dataset, user_idx_tensor)
            _, top_k_items = torch.topk(scores, k=top_k)
            final_recommendations[user] = [
                dataset.id2item[i.item()] for i in top_k_items
            ]
    return final_recommendations


# 추천을 생성하고 CSV 파일로 저장
def save_to_csv(recommendations, model_name, metric_value, path="recommendations"):
    # 파일 이름 포맷을 '모델이름_메트릭결과값.csv' 형태로 설정
    formatted_path = f"{path}_{model_name}_{metric_value:.4f}.csv"
    with open(formatted_path, "w") as f:
        f.write("user,item\n")
        for user, items in recommendations.items():
            for item in items:
                f.write(f"{user},{item}\n")
    print(f"Recommendations saved to {formatted_path}")


# 추천 생성 및 제출 파일 저장
user_list = df["user"].unique()
recommendations = generate_recommendations(trained_model, dataset, user_list, top_k=10)
save_to_csv(recommendations, model_name, metric_value)  # 최종 제출 파일 경로
