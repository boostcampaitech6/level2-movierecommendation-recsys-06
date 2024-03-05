import pandas as pd
import os

# 파일 불러오기
df = pd.read_csv("../data/train/train_ratings.csv")
output_dir="data/"

# 각 유저 별로 아이템을 3개씩 분할하여 저장
for user_id, group_data in df.groupby('user'):
    # 유저별 아이템 개수
    num_items = len(group_data)
    # 각 분할 파일에 저장될 아이템 개수
    items_per_split = num_items // 3
    
    for i in range(3):
        start_idx = i * items_per_split
        end_idx = (i + 1) * items_per_split if i < 2 else num_items
        split_data = group_data.iloc[start_idx:end_idx]
        
        # 분할된 데이터를 저장할 파일 경로
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"split_{i+1}.csv")
        
        # 파일이 이미 존재하는지 확인
        if os.path.exists(output_file_path):
            # CSV 파일이 이미 존재하면 해당 파일에 추가
            split_data.to_csv(output_file_path, mode='a', header=False, index=False)
        else:
            # 새로운 CSV 파일로 저장
            split_data.to_csv(output_file_path, index=False)


split_files = ["split_1.csv", "split_2.csv", "split_3.csv"]

# 각 split 파일에 대해 반복하여 행 수 출력
for split_file in split_files:
    # split 파일의 경로
    split_path = os.path.join(output_dir, split_file)
    
    # split 파일을 읽어서 행 수를 카운트
    with open(split_path, 'r') as file:
        num_lines = sum(1 for line in file)
    
    print(f"{split_file}의 데이터 갯수: {num_lines} 행")

print("CSV 파일의 행 수:", len(df))
