import os
from tqdm import tqdm
import pandas as pd
import argparse
from collections import defaultdict
from utils import arg_as_list, save_file


def soft_voting(args):
    files = os.listdir(args.file_path)
    files.sort()

    df_list = [pd.read_csv(os.path.join(args.file_path, i)) for i in files]
    user_list = df_list[0]["user"].unique()
    df_len = len(df_list)
    ensemble_ratio = arg_as_list(args.weight)

    result = []
    tbar = tqdm(user_list, desc="Ensemble")

    for user in tbar:
        temp = defaultdict(float)
        for idx in range(df_len):
            items = df_list[idx][df_list[idx]["user"] == user]["item"].values

            for item_idx, item in enumerate(items):
                temp[item] += ensemble_ratio[idx] * (1 - item_idx / len(items))

        for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:10]:
            result.append((user, key))

    output = pd.DataFrame(result, columns=["user", "item"])

    save_file(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default="./input/")
    parser.add_argument("--weight", type=str, default="0.5,0.5")

    args = parser.parse_args()
    soft_voting(args)