import os
import sys
import re
import MeCab
from bs4 import BeautifulSoup
import kss
import json
import numpy as np 
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
DATA_DIR = PROJECT_DIR + '/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
MODEL_DIR = PROJECT_DIR + '/models'  
LOG_FILE = PROJECT_DIR + '/logs/preprocessing.log' # logs -> for storing logs information during preprocess and finetuning

def korean_tokenizer(text, use_tags=None, print_tag=False): 
    tokenizer = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ko-dic")
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    result = []
    
    if use_tags:
        for word_ in word_tag[:-2]:
            word = word_.split("\t")
            tag = word[1].split(",")[0]

            if(tag in use_tags):     
                if print_tag:
                    result.append((word[0], tag))
                else:
                    result.append(word[0]) 
    else:
        for word_ in word_tag[:-2]:
            word = word_.split("\t")
            result.append(word[0]) 

    return result

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
    url_pattern = re.compile(r'https?://\S*|www\.\S*')
    text = url_pattern.sub(r'URL', text)

    # html 삭제
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 숫자 중간에 공백 삽입하기
    text = number_split(text)
    #number_pattern = re.compile('\w*\d\w*') 
#     number_pattern = re.compile('\d+') 
#     text = number_pattern.sub(r'[[NUMBER]]', text)
    

    # PUCTUACTION_TO_REMOVED = string.punctuation.translate(str.maketrans('', '', '\"\'#$%&\\@'))  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 중 적은것을 제외한 나머지를 삭제
    # text = text.translate(str.maketrans(PUCTUACTION_TO_REMOVED, ' '*len(PUCTUACTION_TO_REMOVED))) 

    # remove_redundant_white_spaces
    text = re.sub(' +', ' ', text)
    
    return text

def preprocessing(text, tokenizer=None):
    text_p = noise_remove(text)
    if tokenizer is not None:
        text_p = tokenizer(text_p)
        text_p = ' '.join(text_p)

    return text_p

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


def create_json_files(df, type, target_summary_sent='abs', path=''):
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

        file_name = os.path.join(f'{path}/{type}_abs',f"{type}.{start_idx_str}_{end_idx_str}.json")
        
        json_list = []
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [preprocessing(original_sent, korean_tokenizer).split()
                                    for original_sent in row['article_original']]
            # original_sents_list = [preprocessing(original_sent).split()
            #                         for original_sent in row['article_original']]

            if target_summary_sent == 'ext':
                summary_sents = row['extractive_sents']
            elif target_summary_sent == 'abs':
                summary_sents = korean_sent_spliter(row['abstractive'])   

            summary_sents_list = [preprocessing(original_sent, korean_tokenizer).split()
                                    for original_sent in summary_sents] if type == 'train' else []
            # summary_sents_list = [preprocessing(original_sent).split()
            #                         for original_sent in summary_sents] if type == 'train' else []

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


# python make_json_data.py train abs
if __name__ == '__main__':

    if sys.argv[1] == 'train': # valid
        ## make json file
        with open(RAW_DATA_DIR + '/ext/train.jsonl', 'r') as json_file:
            json_list = list(json_file)

        trains = []
        for json_str in json_list:
            line = json.loads(json_str)
            trains.append(line)

        df = pd.DataFrame(trains)
        df['extractive_sents'] = df.apply(lambda row: list(np.array(row['article_original'])[row['extractive']]) , axis=1)

        # randomly split
        train_df = df.sample(frac=0.95,random_state=42) #random state is a seed value
        valid_df = df.drop(train_df.index)
        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        create_json_files(train_df, type='train', target_summary_sent=sys.argv[2], path=JSON_DATA_DIR)
        create_json_files(valid_df, type='valid', target_summary_sent=sys.argv[2], path=JSON_DATA_DIR)

        
        ## make bert data
        
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 넘어감
        os.system(f"python preprocess.py \
                -mode format_to_bert -dataset train\
                -raw_path {JSON_DATA_DIR}/train_{sys.argv[2]} -save_path {BERT_DATA_DIR}/train_{sys.argv[2]} -log_file {LOG_FILE} \
                -lower -n_cpus 24")
        os.system(f"python preprocess.py \
                -mode format_to_bert -dataset valid \
                -raw_path {JSON_DATA_DIR}/valid_{sys.argv[2]} -save_path {BERT_DATA_DIR}/valid_{sys.argv[2]} -log_file {LOG_FILE} \
                -lower -n_cpus 24")


    elif sys.argv[1] == 'test':
        with open(RAW_DATA_DIR + '/ext/extractive_test_v2.jsonl', 'r') as json_file:
            json_list = list(json_file)

        tests = []
        for json_str in json_list:
            line = json.loads(json_str)
            tests.append(line)

        test_df = pd.DataFrame(tests)
        create_json_files(test_df, type='test', path=JSON_DATA_DIR)


        # !python preprocess.py \
        #     -mode format_to_bert -dataset test \
        #     -raw_path $JSON_DATA_DIR -save_path $BERT_DATA_DIR -log_file $LOG_FILE \
        #     -lower -n_cpus 8
                        