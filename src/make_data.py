import os
import sys
import re
#import MeCab
from bs4 import BeautifulSoup
import kss
import json
import numpy as np 
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import ast
import argparse
import pickle

PROBLEM = 'ext'

## 사용할 path 정의
# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
PROJECT_DIR = '..'

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 


# special_symbols_in_dict = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-']
# unused_tags = ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']
# def korean_tokenizer(text, unused_tags=None, print_tag=False): 
#     # assert if use_tags is None or unuse_tags is None
    
#     tokenizer = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ko-dic")
#     parsed = tokenizer.parse(text)
#     word_tag = [w for w in parsed.split("\n")]
#     result = []
    
#     if unused_tags:
#         for word_ in word_tag[:-2]:
#             word = word_.split("\t")
#             tag = word[1].split(",")[0]
#             if tag not in unused_tags:
#                 if print_tag:
#                     result.append((word[0], tag))
#                 else:
#                     result.append(word[0]) 
#     else:
#         for word_ in word_tag[:-2]:
#             word = word_.split("\t")
#             result.append(word[0]) 

#     return result

def number_split(sentence):
    # 1. 공백 이후 숫자로 시작하는 경우만(문자+숫자+문자, 문자+숫자 케이스는 제외), 해당 숫자와 그 뒤 문자를 분리
    num_str_pattern = re.compile(r'(\s\d+)([^\d\s])')
    sentence = re.sub(num_str_pattern, r'\1 \2', sentence)

    # 2. 공백으로 sentence를 분리 후 숫자인경우만 공백 넣어주기
    #numbers_reg = re.compile("\s\d{2,}\s")
    sentence_fixed = ''
    for token in sentence.split():
        if token.isnumeric():
            token = ' '.join(token)
        sentence_fixed+=' '+token
    return sentence_fixed

def noise_remove(text):
    text = text.lower()
    
    # url 대체
    # url_pattern = re.compile(r'https?://\S*|www\.\S*')
    # text = url_pattern.sub(r'URL', text)

    # html 삭제
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 숫자 중간에 공백 삽입하기
    # text = number_split(text)
    #number_pattern = re.compile('\w*\d\w*') 
#     number_pattern = re.compile('\d+') 
#     text = number_pattern.sub(r'[[NUMBER]]', text)
    

    # PUCTUACTION_TO_REMOVED = string.punctuation.translate(str.maketrans('', '', '\"\'#$%&\\@'))  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 중 적은것을 제외한 나머지를 삭제
    # text = text.translate(str.maketrans(PUCTUACTION_TO_REMOVED, ' '*len(PUCTUACTION_TO_REMOVED))) 

    # remove_redundant_white_spaces
    text = re.sub(' +', ' ', text)

    # tgt special token 으로 활용할 204; 314[ 315] 대체/삭제해줘서 없애주기
    text = re.sub('¶', ' ', text)
    text = re.sub('----------------', ' ', text)
    text = re.sub(';', '.', text)

    return text

def preprocessing(text, tokenizer=None):
    text = noise_remove(text)
    if tokenizer is not None:
        text = tokenizer(text)
        text = ' '.join(text)

    return text

def korean_sent_spliter(doc):
    sents_splited = kss.split_sentences(doc)
    if len(sents_splited) == 1:
        # .이나 ?가 있는데도 kss가 분리하지 않은 문장들을 혹시나해서 살펴보니
        # 대부분 쉼표나 가운데점 대신 .을 사용하거나 "" 사이 인용문구 안에 들어가있는 점들. -> 괜찮.
        # aa = sents_splited[0].split('. ')
        # if len(aa) > 1:
        #     print(sents_splited)
        return sents_splited
    else:  # kss로 분리가 된 경우(3문장 이상일 때도 고려)
        #print(sents_splited)
        for i in range(len(sents_splited) - 1):
            idx = 0
            # 두 문장 사이에 .이나 ?가 없는 경우: 그냥 붙여주기
            if sents_splited[idx][-1] not in ['.','?' ] and idx < len(sents_splited) - 1:
                sents_splited[idx] = sents_splited[idx] + ' ' + sents_splited[idx + 1] if doc[len(sents_splited[0])] == ' ' \
                                        else sents_splited[idx] + sents_splited[idx + 1] 
                del sents_splited[idx + 1]
                idx -= 1
        #print(sents_splited)
        return sents_splited


def create_json_files(df, data_type='train', target_summary_sent=None, path=''):
    NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  # -1로 하니 안됨...

        #정렬을 위해 앞에 0 채워주기
        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx)))*'0' + str(start_idx)
        end_idx_str = (length - len(str(end_idx-1)))*'0' + str(end_idx-1)

        file_name = os.path.join(f'{path}/{data_type}_{target_summary_sent}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json') if target_summary_sent is not None \
                    else os.path.join(f'{path}/{data_type}' \
                                + f'/{data_type}.{start_idx_str}_{end_idx_str}.json')
        
        json_list = []
        for i, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [preprocessing(original_sent).split() for original_sent in row['article_original']]
            summary_sents_list = []
            if target_summary_sent is not None:
                if target_summary_sent == 'ext':
                    summary_sents = ast.literal_eval(row['extractive_sents'])
                elif target_summary_sent == 'abs':
                    summary_sents = korean_sent_spliter(row['abstractive'])
                summary_sents_list = [preprocessing(original_sent).split() for original_sent in summary_sents]
            json_list.append({'src': original_sents_list,
                              'tgt': summary_sents_list
            })
        #     print(json_list)
        #     break
        # break
        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        #print(json_string)
        with open(file_name, 'w') as json_file:
            json_file.write(json_string)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default=None, type=str, choices=['df', 'train_bert', 'test_bert'])
    parser.add_argument("-target_summary_sent", default='ext', type=str)
    parser.add_argument("-n_cpus", default='2', type=str)

    args = parser.parse_args()

    # python make_data.py -make df
    # Convert raw data to df
    if args.task == 'df': # and valid_df
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

        # import data
        with open(f'{RAW_DATA_DIR}/train.jsonl', 'r') as json_file:
            train_json_list = list(json_file)
        with open(f'{RAW_DATA_DIR}/test.jsonl', 'r') as json_file:
            test_json_list = list(json_file)

        trains = []
        for json_str in train_json_list:
            line = json.loads(json_str)
            trains.append(line)
        tests = []
        for json_str in test_json_list:
            line = json.loads(json_str)
            tests.append(line)

        # Convert raw data to df
        df = pd.DataFrame(trains)
        import ast
        #df['extractive'] = df['extractive'].apply(lambda x: json.loads(x))
        #df['article_original'] = df['article_original'].apply(lambda x: ast.literal_eval(x))

        new_df = []
        for i in trange(len(df)):
            df_len = len(df.iloc[i]['extractive'])
            df_text = []
            for j in range(df_len):
                idx = df.iloc[i]['extractive'][j]
                if idx is None:
                    pass
                else:
                    df_text.append(df.iloc[i]['article_original'][idx])
            new_df.append(str(df_text))
        df = pd.concat([df, pd.DataFrame(new_df, columns=['extractive_sents'])], axis=1)
        #df['extractive_sents'] = df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)

        # random split
        train_df = df.sample(frac=0.95,random_state=42) #random state is a seed value
        valid_df = df.drop(train_df.index)
        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        test_df = pd.DataFrame(tests)

        # save df
        train_df.to_pickle(f"{RAW_DATA_DIR}/train_df.pickle")
        valid_df.to_pickle(f"{RAW_DATA_DIR}/valid_df.pickle")
        test_df.to_pickle(f"{RAW_DATA_DIR}/test_df.pickle")
        print(f'train_df({len(train_df)}) is exported')
        print(f'valid_df({len(valid_df)}) is exported')
        print(f'test_df({len(test_df)}) is exported')
        
    # python make_data.py -make bert -by abs
    # Make bert input file for train and valid from df file
    elif args.task  == 'train_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        for data_type in ['train', 'valid']:
            df = pd.read_pickle(f"{RAW_DATA_DIR}/{data_type}_df.pickle")
            ## make json file
            # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
            json_data_dir = f"{JSON_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(json_data_dir):
                os.system(f"rm {json_data_dir}/*")
            else:
                os.mkdir(json_data_dir)

            create_json_files(df, data_type=data_type, target_summary_sent=args.target_summary_sent, path=JSON_DATA_DIR)

            ## Convert json to bert.pt files
            bert_data_dir = f"{BERT_DATA_DIR}/{data_type}_{args.target_summary_sent}"
            if os.path.exists(bert_data_dir):
                os.system(f"rm {bert_data_dir}/*")
            else:
                os.mkdir(bert_data_dir)
            
            os.system(f"python preprocess.py"
                + f" -mode format_to_bert -dataset {data_type}"
                + f" -raw_path {json_data_dir}"
                + f" -save_path {bert_data_dir}"
                + f" -log_file {LOG_PREPO_FILE}"
                + f" -lower -n_cpus {args.n_cpus}")


    # python make_data.py -task test_bert
    # Make bert input file for test from df file
    elif args.task  == 'test_bert':
        os.makedirs(JSON_DATA_DIR, exist_ok=True)
        os.makedirs(BERT_DATA_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        test_df = pd.read_pickle(f"{RAW_DATA_DIR}/test_df.pickle")

        ## make json file
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
        json_data_dir = f"{JSON_DATA_DIR}/test"
        if os.path.exists(json_data_dir):
            os.system(f"rm {json_data_dir}/*")
        else:
            os.mkdir(json_data_dir)

        create_json_files(test_df, data_type='test', path=JSON_DATA_DIR)
        
        ## Convert json to bert.pt files
        bert_data_dir = f"{BERT_DATA_DIR}/test"
        if os.path.exists(bert_data_dir):
            os.system(f"rm {bert_data_dir}/*")
        else:
            os.mkdir(bert_data_dir)
        
        os.system(f"python preprocess.py"
            + f" -mode format_to_bert -dataset test"
            + f" -raw_path {json_data_dir}"
            + f" -save_path {bert_data_dir}"
            + f" -log_file {LOG_PREPO_FILE}"
            + f" -lower -n_cpus {args.n_cpus}")
