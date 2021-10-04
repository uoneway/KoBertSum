

# KoBertSum

## Update [21.09.]
- 다양한 데이터를 받아들일 수 있도록 `Data Preparation` 단계 관련 코드를 전면 수정하였습니다.
- hydra를 적용하여 많은 arguments들을 쉽게 관리 및 수정할 수 있도록 하였습니다.

  
추후 업데이트 계획은 다음과 같습니다.
- [ ] `BertSumAbs` 및 `BertSumExtAbs` 요약모델 추가 지원
- [ ] Pre-trained BERT로 [KoBERT ](https://github.com/SKTBrain/KoBERT)외 타 모델 지원(Huggingface transformers 라이브러리 지원 모델 위주)

## 모델 소개

### KoBertSum이란?

KoBERTSUM은 ext 및 abs summarizatoin 분야에서 우수한 성능을 보여주고 있는 [BertSum모델](https://github.com/nlpyang/PreSumm)을 한국어 데이터에 적용할 수 있도록 수정한 한국어 요약 모델입니다.

- Pre-trained BERT로 [kykim/bert-kor-base](https://huggingface.co/kykim/bert-kor-base/blob/main/config.json), [monologg/kobert](https://huggingface.co/monologg/kobert/blob/main/config.json)를 지원합니다.
- 다양한 데이터 형태에 대한 옵션값을 지원하여 이용자가 원하는 데이터를 쉽게 가공하여 학습시킬수 있습니다.
- hydra를 적용하여 다양한 arguments에 대한 체계적 실험이 용이합니다.


### BertSum이란?

BertSum은 BERT 위에 inter-sentence Transformer 2-layers 를 얹은 구조를 갖습니다. 이를 fine-tuning하여 extract summarization을 수행하는 `BertSumExt`, abstract summarization task를 수행하는 `BertSumAbs` 및 `BertSumExtAbs` 요약모델을 포함하고 있습니다.

- 논문:  [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (EMNLP 2019 paper)
- 원코드: https://github.com/nlpyang/PreSumm

기 Pre-trained BERT를 summarization task 수행을 위한 embedding layer로 활용하기 위해서는 여러 sentence를 하나의 인풋으로 넣어주고, 각 sentence에 대한 정보를 출력할 수 있도록 입력을 수정해줘야 합니다. 이를 위해

- Input document에서 매 문장의 앞에 [CLS] 토큰을 삽입하고
    ( [CLS] 토큰에 대응하는 BERT 결과값(T[CLS])을 각 문장별 representation으로 간주)

- 매 sentence마다 다른 segment embeddings 토큰을 더해주는 interval segment embeddings을 추가합니다.

  ![BERTSUM_structure](tutorials/images/BERTSUM_structure.PNG)


## Usage
- 전체 실행 코드 예시는 `tutorials/turorial.ipynb`를 참고하세요 
- Data 및 폴더 구성을 규칙대로 만들어주셔야 합니다. `datasets/sample`를 참고하세요.
### Data Preparation

1. 필요 라이브러리 설치하기
    ```
    pip install -r requirements.txt
    pip install -r requirements_prepro.txt
    ```

2. dataframe을 pickle로 저장한 `.pkl` 데이터 만들기
  - 본 라이브러리는 dataframe을 pickle로 저장한 `.pkl` 데이터를 입력으로 받습니다. 
  - `.pkl`로 저장한 dataframe 데이터를, `datasets/DATASET_NAME/df` 폴더에 넣어줍니다.
  - 이 떄 파일명에 `train`(필요시 `valid`도), `test` 텍스트가 포함되어 있는지에 따라 데이터 종류를 구분하고 그에 맞게 처리합니다. 따라서 파일명에 해당 키워드가 포함되도록 파일명을 저장해주세요.
  - 당연히 `train`(필요시 `valid`도) 데이터는 본문뿐만 아니라 summary 데이터를 포함하고 있어야 합니다.
    ```
    train_df.to_pickle(PROJECT_DIR + "/datasets/sample/df/train_df.pkl")
    valid_df.to_pickle(PROJECT_DIR + "/datasets/sample/df/valid_df.pkl")
    test_df.to_pickle(PROJECT_DIR + "/datasets/sample/df/test_df.pkl")
    ```
  
3. 다음 명령어를 실행하여 위에서 저장한 데이터를 BertSum 모델에 입력 가능한 `.pt` 파일로 변환   
변환이 완료된 데이터는 `datasets/DATASET_NAME/bert`에 저장됩니다.
    ```
    # cd src
    python make_data.py \
      dataset=sample \
      dataset.text.name=text dataset.text.type=str_list dataset.text.do_cleaning=True \
      dataset.summary.name=summary dataset.summary.type=str \
      dataset.is_informal=False \
      model_name="kykim/bert-kor-base" n_cpus=28
    ```
     - `dataset`: 데이터셋 이름 (위 2번 단계에서 만들어준 DATASET_NAME과 동일해야 함)
     - `dataset.text.name`: 데이터에서 본문에 해당하는 colname(df)
     - `dataset.text.type`: 데이터에서 본문이 list of string 인지(`str_list`), 아니면 string인지(`str`)
      만약 string이라면 내부에서 이를 문장 단위로 잘르는 작업까지 수행합니다.
     - `dataset.text.do_cleaning`: 기본적인 데이터 클리닝을 적용할지 여부를 선택(`src/prepro/preprocessor_kr.py` 내 `pre_remove_noise` 참고)
     - `dataset.summary.name`: 데이터에서 요약문에 해당하는 colname(df)
     - `dataset.summary.type`: 데이터 내 요약문이 index list형태로 되어있는지(`idx_list`, 즉 본문의 sentence index list), string list로 되어 있는지(`str_list`), 아니면 string인지(`str`)
     - `model_name`: 이용할 한글 Bert model 이름. [kykim/bert-kor-base](https://huggingface.co/kykim/bert-kor-base/blob/main/config.json) 또는 [monologg/kobert](https://huggingface.co/monologg/kobert/blob/main/config.json)
     - `n_cpus`(default=16): 연산에 이용할 CPU 수

    참고로 본 라이브러리는 hydra를 이용하여 arguments를 쉽게 관리할 수 있도록 구성하였습니다.(`conf/make_data` 폴더 챰고)   
    현재 다음의 데이터셋에 대해 지원하고 있으며, 이는 단 몇 줄의 명령어로 Data Preparation를 끝낼 수 있음을 의미합니다.
    - [한국어 문서 추출요약 AI 경진대회](https://dacon.io/competitions/official/235671/data/) 요약 데이터셋: `bflysoft_ko`
      1. 해당 `.jsonl`파일을 다운받아 df로 불러온 후 `datasets/bflysoft_ko/df` 폴더에 `train_df.pkl`과 같은 이름으로 저장해줍니다.
      2. `python make_data.py dataset=bflysoft_ko` 를 실행합니다.
    - [훈민정음에 스며들다"-문서요약 역량평가](https://dacon.io/competitions/official/235818/data) 요약 데이터 셋: `hunmin_ko`
  
### Fine-tuning and Inference

1. 필요 라이브러리 설치하기
    ```
    pip install -r requirements.txt
    ```
2. Fine-tuning

    KoBERT 모델을 기반으로 fine-tuning을 진행하고, 1,000 step마다  Fine-tuned model 파일(`.pt`)을 저장합니다. 

    - `target_summary_sent`: `abs` 또는 `ext` . 
    - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
      예) (GPU 3개를 이용할 경우): `0,1,2`

    ```
    python main.py -mode train -target_summary_sent abs -visible_gpus 0
    ```

    결과는  `models` 폴더 내 finetuning이 실행된 시간을 폴더명으로 가진 폴더에 저장됩니다. 

3. Validation

   Fine-tuned model마다 validation data set을 통해 inference를 시행하고, loss 값을 확인합니다.

   - `model_path`:  model 파일(`.pt`)이 저장된 폴더 경로

   ```
   python main.py -mode valid -model_path 1209_1236
   ```

   결과는 `ext/logs` 폴더 내 `valid_1209_1236.log` 형태로 저장됩니다.

4. Inference & make submission file

    Validation을 통해 확인한 가장 성능이 우수한 model파일을 통해 실제로 텍스트 요약 과업을 수행합니다.

    - `test_from`:  model 파일(`.pt`) 경로
    - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
      예) (GPU 3개를 이용할 경우): `0,1,2`

    ```
    python main.py -mode test -test_from 1209_1236/model_step_7000.pt -visible_gpus 0
    ```

    결과는 `ext/data/results/` 폴더에 `result_1209_1236_step_7000.candidate`  및 `submission_날짜_시간.csv` 형태로 저장됩니다.