import pandas as pd
import json
import os

def update_item_keys(dataframes, item2attribute):
    new_item2attribute = {}
    item_mapping = {}
    new_item_id = 1
    
    for old_item_id in sorted(map(int, item2attribute.keys())):
        new_item2attribute[str(new_item_id)] = item2attribute[str(old_item_id)]
        item_mapping[str(old_item_id)] = new_item_id
        new_item_id += 1
    
    updated_dataframes = []
    for df in dataframes:
        df['item'] = df['item'].map(lambda x: item_mapping[str(x)])
        updated_dataframes.append(df)
    
    return updated_dataframes, new_item2attribute

# 데이터 불러오기
data_dir = "/data/train/"
directors = pd.read_csv(data_dir + "directors.tsv", delimiter="\t")
genres = pd.read_csv(data_dir + "genres.tsv", delimiter="\t")
titles = pd.read_csv(data_dir + "titles.tsv", delimiter="\t")
writers = pd.read_csv(data_dir + "writers.tsv", delimiter="\t")
years = pd.read_csv(data_dir + "years.tsv", delimiter="\t")
train = pd.read_csv(data_dir + "train_ratings.csv")

# item2attribute 파일 경로 설정
item2attribute_file = data_dir + "Ml_item2attributes.json"

# item2attribute 파일 불러오기
with open(item2attribute_file, "r") as infile:
    item2attribute = json.load(infile)

# 데이터와 item2attribute 수정
dataframes = [directors, genres, titles, writers, years, train]
updated_dataframes, new_item2attribute = update_item_keys(dataframes, item2attribute)

# 수정된 데이터 저장
output_dir = "data/ntrain/"
os.makedirs(output_dir, exist_ok=True)

for i, df in enumerate(updated_dataframes):
    if i == 5:  # train_ratings.csv
        df.to_csv(output_dir + "train_ratings.csv", index=False)  # train_ratings.csv는 index를 저장하지 않음
    else:
        df.to_csv(output_dir + ["directors.tsv", "genres.tsv", "titles.tsv", "writers.tsv", "years.tsv"][i], sep="\t", index=False)
        
# 수정된 item2attribute 딕셔너리를 JSON 파일로 저장
with open(output_dir + "Ml_item2attributes.json", "w") as outfile:
    json.dump(new_item2attribute, outfile)
