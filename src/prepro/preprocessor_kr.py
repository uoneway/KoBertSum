import re
#import MeCab
from bs4 import BeautifulSoup
import kss

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

def preprocess_kr(text, tokenizer=None):
    text = noise_remove(text)
    if tokenizer is not None:
        text = tokenizer(text)
        text = ' '.join(text)

    return text

def korean_sent_spliter(doc):
    sents_splited = kss.split_sentences(doc, safe=True)
    return sents_splited
    # if len(sents_splited) == 1:
    #     # .이나 ?가 있는데도 kss가 분리하지 않은 문장들을 혹시나해서 살펴보니
    #     # 대부분 쉼표나 가운데점 대신 .을 사용하거나 "" 사이 인용문구 안에 들어가있는 점들. -> 괜찮.
    #     # aa = sents_splited[0].split('. ')
    #     # if len(aa) > 1:
    #     #     print(sents_splited)
    #     return sents_splited
    # else:  # kss로 분리가 된 경우(3문장 이상일 때도 고려)
    #     #print(sents_splited)
    #     for i in range(len(sents_splited) - 1):
    #         idx = 0
    #         # 두 문장 사이에 .이나 ?가 없는 경우: 그냥 붙여주기
    #         if sents_splited[idx][-1] not in ['.','?' ] and idx < len(sents_splited) - 1:
    #             sents_splited[idx] = sents_splited[idx] + ' ' + sents_splited[idx + 1] if doc[len(sents_splited[0])] == ' ' \
    #                                     else sents_splited[idx] + sents_splited[idx + 1] 
    #             del sents_splited[idx + 1]
    #             idx -= 1
    #     #print(sents_splited)
    #     return sents_splited