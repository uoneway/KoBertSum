import os
import sys
import time
from src.others.test_rouge_score import RougeScorer

PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
DATA_DIR = PROJECT_DIR + '/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 

MODEL_DIR = PROJECT_DIR + '/models'  
LOG_DIR = PROJECT_DIR + '/logs' # logs -> for storing logs information during preprocess and finetuning
RESULT_DIR = PROJECT_DIR + '/results' 

if __name__ == '__main__':
    os.chdir(PROJECT_DIR + '/src')
    now = time.strftime('%y%m%d_%H%M')

    if sys.argv[1] == 'data':
        ### 전처리
        os.system("python make_json_data.py train abs")
        os.system("python make_json_data.py test")

        os.system("python make_bert_data.py train")
        os.system("python make_bert_data.py test")

    elif sys.argv[1] == 'train':
        os.system(f'mkdir {MODEL_DIR}/{now}')

        """
         파라미터
        ext_dropout: 
        """

        # python train.py -task ext -mode train -bert_data_path BERT_DATA_PATH -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
        # !python train.py  -task abs -mode train -train_from /kaggle/input/absbert-weights/model_step_149000.pt -bert_data_path /kaggle/working/bert_data/news  -dec_dropout 0.2  -model_path /kaggle/working/bertsumextabs -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 150000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0  -temp_dir /kaggle/working/temp -log_file /kaggle/working/logs/abs_bert_cnndm
        parameters_1800 = "-ext_dropout 0.1 -lr 2e-3 -batch_size 500 -train_steps 5000 -accum_count 2 -use_interval true -warmup_steps 3000 -max_pos 512"
        parameters_201203_0300 = "-ext_dropout 0.1 -lr 2e-3 -batch_size 1000 -train_steps 5000 -accum_count 2 -use_interval true -warmup_steps 3000 -max_pos 512"
        parameters_201204_0300 = "-ext_dropout 0.1 -lr 2e-3 -batch_size 3000 -train_steps 50000 -accum_count 2 -use_interval true -warmup_steps 3000 -max_pos 512"

        os.system(f"""
        python train.py -task ext -mode train \
            -bert_data_path {BERT_DATA_DIR}/train_abs -model_path {MODEL_DIR}/{now} -log_file {LOG_DIR}/train_{now}.log \
            -visible_gpus 0,1 -report_every 50 \
            -save_checkpoint_steps 200 \
            {parameters_201204_0300}
        """)

    elif sys.argv[1] == 'test':
        os.system(f"""
        python train.py -task ext -mode test \
            -test_from {MODEL_DIR}/{sys.argv[2]} \
            -bert_data_path {BERT_DATA_DIR}/test \
            -result_path {RESULT_DIR}/result -log_file {LOG_DIR}/test_{now}.log \
            -model_path {MODEL_DIR} \
            -test_batch_size 1  -batch_size 300 \
            -sep_optim true -use_interval true -visible_gpus 0,1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 \
            -report_rouge False  \
        """)
        #  -model_path {MODEL_DIR} 

    elif sys.argv[1] == 'rouge':
        pass
        # rouge_scorer = RougeScorer()
        # str_scores = rouge_scorer.compute_rouge(ref_df, hyp_df)
        # rouge_scorer.save_rouge_scores(str_scores)
        # rouge_scorer.format_rouge_scores(rouge_scorer.scores)

