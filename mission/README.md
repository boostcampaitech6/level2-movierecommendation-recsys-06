# py 파일 구조

### - 전처리
1. modules_for_preprocessing.py
2. preprocess.py

### - 모델 학습 및 추론
1. dataloader.py
2. metrics.py
3. models.py
4. run.py
5. trainer.py
6. main.py



# 활용 방법

### 1. 사전에 전처리를 반드시 수행!
python preprocess.py
: 전처리에서 2가지 버전의 data split이 이루어집니다
- pro_sg: train(80) | val(10) | test(10)
- pro_sg2: train(90) | val(10)
- json_id: id2~.json | id2~_2.json

### 2. 이후 모델 학습 가능!
8:1:1 split: python main.py --model 모델명
9:1 split: python main2.py --model 모델명

    1) **제출파일**
        mission>submission에 저장됨
        파일 명: {모델 명} {실행시간}.csv
        
    2) **모델파일**
        mission>model_files에 저장됨
        파일 명: {모델 명} {실행시간}.pt


# 모델 종류
- MultiDAE
    python main.py --model MultiDAE
- MultiVAE
    python main.py --model MultiVAE
- RecVAE
    python main.py --model RecVAE
- EASE
    python main.py --model EASE