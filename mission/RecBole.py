# 단계 1: RecBole 설치
# Python 환경에서 pip를 사용하여 RecBole을 설치할 수 있습니다:
# !pip install recbole

# 단계 2: 데이터셋 준비
# RecBole은 일반적으로 user_id, item_id, timestamp 열을 포함하는 데이터셋 형식을 요구합니다.
# 추가적인 특성이 있으면 포함할 수도 있습니다. 데이터셋을 이 열들을 포함하는 .csv 형식으로 준비하세요.

# 예시: 필요한 열을 포함하여 'interaction.csv'와 'item.csv'로 데이터셋을 저장합니다.
# interaction.csv: user_id, item_id, timestamp
# item.csv: item_id, director, genre, title 등과 같은 특성 열들

# 단계 3: 모델 및 매개변수 설정
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

# 모델을 위한 설정을 정의합니다
config_dict = {
    "data_path": "./your_data_directory",  # .csv 파일을 포함하는 디렉토리 경로
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "TIME_FIELD": "timestamp",
    "NEG_SAMPLING": None,  # 데이터가 부정적인 샘플을 포함하지 않는 암시적인 경우 None으로 설정
    # 학습률, 에포크 등과 같은 추가 설정을 여기에 추가할 수 있습니다
}
config = Config(model="BPR", dataset="your_dataset_name", config_dict=config_dict)

# 단계 4: 모델 학습
# 재현성을 위한 랜덤 시드 초기화
init_seed(config["seed"], config["reproducibility"])

# 출력을 위한 로거 초기화
logger = init_logger(config)

# 학습 및 평가를 위한 데이터셋과 데이터 로더 생성
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# 모델 초기화
model = BPR(config, train_data.dataset).to(config["device"])

# 트레이너 초기화
trainer = Trainer(config, model)

# 모델 학습
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True)

# 단계 5: 모델 평가
# 테스트 데이터셋에서 학습된 모델 평가
test_result = trainer.evaluate(test_data)

# 평가 결과 출력
print(f"테스트 결과: {test_result}")

# 필요한 경우 모델 저장
# trainer.save(model_path='./saved_models/')
