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

### 2. 이후 모델 학습 가능!
python main.py

    1) 제출파일\
        mission>submission에 저장됨\
        파일 명: {모델 명} {실해시간}.csv\
    2) 모델파일\
        mission>model_files에 저장됨\
        파일 명: {모델 명} {실행시간}.pt


# 유의사항
현재는 static model의 전처리만 지원\
차후에 sequential model도 적용 예정
