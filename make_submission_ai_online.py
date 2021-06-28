import json
import numpy as np
import pandas as pd
import time
import re
import sys
import os

PROBLEM = 'ext'

## 사용할 path 정의
# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
PROJECT_DIR = os.getcwd() # '..'
print(PROJECT_DIR)

# DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
DATA_DIR = f'{PROJECT_DIR}/datasets/ai_online'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
# BERT_DATA_DIR = DATA_DIR + '/bert_data' 
BERT_DATA_DIR = DATA_DIR + '/bert' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

MODEL_DIR = f'{PROJECT_DIR}/{PROBLEM}/models' 
RESULT_DIR = f'{PROJECT_DIR}/{PROBLEM}/results' 

# python make_submission.py result_1209_1236_step_7000.candidate
if __name__ == '__main__':
    # # test set
    # with open(RAW_DATA_DIR + '/extractive_test_v2.jsonl', 'r') as json_file:
    #     json_list = list(json_file)

    # tests = []
    # for json_str in json_list:
    #     line = json.loads(json_str)
    #     tests.append(line)
    # test_df = pd.DataFrame(tests)
    
    def load_json(path):
        return pd.read_json(path, orient='records', encoding='utf-8-sig')
    # print(os.path.join(DATA_DIR, 'orig', 'test.json'))
    test_df = load_json(os.path.join(DATA_DIR, 'orig', 'test.json'))

    # 추론결과
    with open(RESULT_DIR + '/' + sys.argv[1], 'r') as file:
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    for line in lines:
        sum_sents_text, sum_sents_idxes = line.rsplit(r'[', maxsplit=1)
        sum_sents_text = sum_sents_text.replace('<q>', '\n')
        sum_sents_idx_list = [ int(str.strip(i)) for i in sum_sents_idxes[:-2].split(', ')]
        print(sum_sents_idx_list)
        test_pred_list.append({'sum_sents_tokenized': sum_sents_text, 
                            'summary_index1': sum_sents_idx_list[0],
                            'summary_index2': sum_sents_idx_list[1],
                            'summary_index3': sum_sents_idx_list[2],
                            })

    # [{"ID":79095,"summary_index1":0,"summary_index2":1,"summary_index3":2},
    
    result_df = pd.merge(test_df, pd.DataFrame(test_pred_list), how="left", left_index=True, right_index=True)
    print(result_df)
    
    result_df['ID'] = result_df['id'].astype(int)
    # result_df['ID'] = result_df['id'].astype(int)
    result_df.drop(['id', 'article_original', 'sum_sents_tokenized'], axis=1, inplace=True)
    
    # export
    now = time.strftime('%y%m%d_%H%M')
    result_df[["ID", "summary_index1", "summary_index2", "summary_index3"]].to_json(f'{RESULT_DIR}/submission_{now}.json', orient="records")
    
    # result_df['summary'] = result_df.apply(lambda row: '\n'.join(list(np.array(row['article_original'])[row['summary_index1']])) , axis=1)
    # submit_df = pd.read_csv(RAW_DATA_DIR + '/extractive_sample_submission_v2.csv')
    # submit_df.drop(['summary'], axis=1, inplace=True)

    # print(result_df['id'].dtypes)
    # print(submit_df.dtypes)

    # result_df['id'] = result_df['id'].astype(int)
    # print(result_df['id'].dtypes)

    # submit_df  = pd.merge(submit_df, result_df.loc[:, ['id', 'summary']], how="left", left_on="id", right_on="id")
    # print(submit_df.isnull().sum())

    # ## 결과 통계치 보기
    # # word
    # abstractive_word_counts = submit_df['summary'].apply(lambda x:len(re.split('\s', x)))
    # print(abstractive_word_counts.describe())

    # # export
    # now = time.strftime('%y%m%d_%H%M')
    # submit_df.to_csv(f'{RESULT_DIR}/submission_{now}.csv', index=False, encoding="utf-8-sig")