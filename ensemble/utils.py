import os 
import pandas as pd 
import numpy as np
from itertools import combinations
import argparse

from datetime import datetime
import pytz

def korea_date_time():
    korea_timezone = pytz.timezone("Asia/Seoul")
    date_time = datetime.now(tz=korea_timezone)
    date_time = date_time.strftime("%Y%m%d_%H%M%S")

    return date_time



def arg_as_list(value):
    # 이 함수는 문자열로 받은 값을 쉼표(,)를 기준으로 분할하여 리스트로 반환합니다.
    return [float(x) for x in value.split(",")]


def save_file(output: pd.DataFrame) -> None:
    # Create the output folder
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    # Save the ensemble result to a CSV file
    date_time = korea_date_time()
    file_name = f"{output_folder}/ensemble {date_time}.csv"
    output.to_csv(file_name, index=False)

    print(f"{file_name} is successfully saved!")