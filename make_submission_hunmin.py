import json
import numpy as np
import pandas as pd
import time
import re
import sys
import os
from src.prepro.preprocessor_kr import post_remove_noise

PROBLEM = "ext"

## 사용할 path 정의'
PROJECT_DIR = os.getcwd()  # '..'
print(PROJECT_DIR)

DATA_DIR = f"{PROJECT_DIR}/datasets/hunmin_ko"
MODEL_DIR = f"{PROJECT_DIR}/{PROBLEM}/models"
RESULT_DIR = f"{PROJECT_DIR}/{PROBLEM}/results"

if __name__ == "__main__":

    # 추론결과
    with open(RESULT_DIR + "/" + sys.argv[1], "r") as file:
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    for idx, line in enumerate(lines):
        sum_sents_text, sum_sents_idxes = line.rsplit(r"[", maxsplit=1)
        sum_sents = sum_sents_text.split("<q>")
        sum = " ".join(sum_sents[:2])
        # sum_sents_idx_list = [int(str.strip(i)) for i in sum_sents_idxes[:-2].split(", ")]
        # print(sum_sents_idx_list)
        test_pred_list.append({"id": idx + 1, "summary": sum})

    result_df = pd.DataFrame(test_pred_list)

    result_df["summary"] = result_df["summary"].apply(post_remove_noise)

    result_df["id"] = result_df["id"].astype(int)
    # result_df['ID'] = result_df['id'].astype(int)
    # result_df.drop(['id', 'article_original', 'sum_sents_tokenized'], axis=1, inplace=True)

    # export
    now = time.strftime("%y%m%d_%H%M")
    result_df[["id", "summary"]].to_csv(f"{RESULT_DIR}/submission_{now}.csv", index=False)
