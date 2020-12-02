import os
import sys
import time

PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
DATA_DIR = PROJECT_DIR + '/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
MODEL_DIR = PROJECT_DIR + '/models'  
LOG_FILE = PROJECT_DIR + '/logs/preprocessing.log' # logs -> for storing logs information during preprocess and finetuning


if __name__ == '__main__':
    if sys.argv[1] == 'data':
        ### 전처리
        os.chdir(PROJECT_DIR + '/src')
        os.system("python make_json_data.py train abs")
        os.system("python make_json_data.py test")

        os.system("python make_bert_data.py train")
        os.system("python make_bert_data.py test")

    elif sys.argv[1] == 'train':
        ## training
        os.chdir(PROJECT_DIR + '/src')
        now = time.strftime('%y%m%d_%H%M')
        os.system('mkdir ' + now)
        # python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
        # !python train.py  -task abs -mode train -train_from /kaggle/input/absbert-weights/model_step_149000.pt -bert_data_path /kaggle/working/bert_data/news  -dec_dropout 0.2  -model_path /kaggle/working/bertsumextabs -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 150000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0  -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/abs_bert_cnndm
        os.system(f"""
        python train.py -task ext -mode train \
            -bert_data_path {BERT_DATA_DIR}/train_abs -model_path {MODEL_DIR} -log_file {LOG_FILE} \
            -ext_dropout 0.1 -lr 2e-3 -batch_size 500 -train_steps 5000 -accum_count 2 -use_interval true -warmup_steps 3000 -max_pos 512 \
            -visible_gpus 0,1 -report_every 50 \
            -save_checkpoint_steps 200 
        """)

    elif sys.argv[1] == 'test':
        pass
        # !python train.py -task ext -mode test \
        #     -test_from $MODEL_DIR/extract/model_step_4600.pt \
        #     -bert_data_path $BERT_DATA_DIR -model_path $MODEL_DIR -log_file $PreSumm_DIR/logs/val_abs_bert_cnndm \
        #     -test_batch_size 1  -batch_size 300 \
        #     -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path $PreSumm_DIR/logs/abs_bert_cnndm \
        #     -report_rouge False
        #         # -batch_size 300 

