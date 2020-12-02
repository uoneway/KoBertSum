import os
import sys

PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
DATA_DIR = PROJECT_DIR + '/data'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_FILE = PROJECT_DIR + '/logs/preprocessing.log' # logs -> for storing logs information during preprocess and finetuning
MODEL_DIR = PROJECT_DIR + '/models'  

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-task", default='train', type=str, choices=['train', 'test'])
    
    # args = parser.parse_args()
    # init_logger(LOG_FILE)
    # eval('data_builder.'+args.mode + '(args)')

    #os.system("echo Hello from the other side!")
    if sys.argv[1] == 'train':
        # 동일한 파일명 존재하면 덮어쓰는게 아니라 넘어감
        os.system(f"python preprocess.py \
                -mode format_to_bert -dataset train \
                -raw_path {JSON_DATA_DIR}/train_abs -save_path {BERT_DATA_DIR}/train_abs -log_file {LOG_FILE} \
                -lower -n_cpus 1 ")
    elif sys.argv[1] == 'test':
        pass
        # !python preprocess.py \
        #     -mode format_to_bert -dataset test \
        #     -raw_path $JSON_DATA_DIR -save_path $BERT_DATA_DIR -log_file $LOG_FILE \
        #     -lower -n_cpus 1 
                        