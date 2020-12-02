import os
import sys


PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
DATA_DIR = PROJECT_DIR + '/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
MODEL_DIR = PROJECT_DIR + '/models'  
LOG_FILE = PROJECT_DIR + '/logs/preprocessing.log' # logs -> for storing logs information during preprocess and finetuning

### 전처리
os.chdir(PROJECT_DIR + '/src')
os.system("python make_json_data.py train abs")
os.system("python make_json_data.py test")

os.system("python make_bert_data.py train")
os.system("python make_bert_data.py test")

## training
os.system("""
python "train.py" -task ext -mode train \
    -bert_data_path $BERT_DATA_DIR -model_path $MODEL_DIR -log_file $LOG_FILE \
    -ext_dropout 0.1 -lr 2e-3 -batch_size 500 -train_steps 5000 -accum_count 2 -use_interval true -warmup_steps 3000 -max_pos 512
    -visible_gpus 0,1 -report_every 50 
    -save_checkpoint_steps 200 
""")

