import re
import functools
from typing import Union, List
from bs4 import BeautifulSoup

# import mecab


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

chinese2hangul_dict = {
    "甲": "갑",
    "乙": "을",
    "丙": "병",
    "​丁": "정",
    "戊": "무",
    "己": "기",
    "庚": "경",
    "辛": "신",
    "壬": "임",
    "癸": "계",
}


##################
def get_pat2unused_dict(rpat, text, unused_idx):
    result = rpat.findall(text)
    # print(result)
    # if isinstance(result, tuple):
    #     # print(result)
    #     result = result[0]
    return {token: f"[unused{unused_idx}] {idx} " for idx, token in enumerate(list(set(result)))}  # 제{idx}조 "


sep_unused_tok_rpat = [
    re.compile(r"(?:\[\d+\])|(?:[가나다라마바사아자차카타파하]\.\s)"),  # 넘버링 [4] 가.  numbering_rpat
    re.compile(
        r"(?:(?:제*[\d()가나다라마바사아자차카타파하]+[조항호목](?:의\s*\d)*\s*)+)|(?:\d(?:-\d)+)"
    ),  # 법조항 제63조 제1항 제1호 (가)목에. law_number_rpat
    re.compile(r"\b\d+\.(?:\s*\d+\.){,2}"),  # 날짜 1984.6.30. 2002. 12. 18.   date_rpat
]

renaming_exp_rpat = re.compile(r"\([^()]*이하\s*\'([\w\s]+)'\w* 한다\s*\)")
gwal_rpat = re.compile(r"\([^()]*\)")
### 구분도 해줘야 하는 것들: unused20부터 시작
def pre_remove_noise_doc(text):
    for idx, rpat in enumerate(sep_unused_tok_rpat):
        pat2unused_dict = get_pat2unused_dict(rpat, text, 20 + idx)
        # print(pat2unused_dict)
        for k, v in pat2unused_dict.items():
            # print(k)
            text = text.replace(k, v)

    return text


def number_split(sentence):
    # 1. 공백 이후 숫자로 시작하는 경우만(문자+숫자+문자, 문자+숫자 케이스는 제외), 해당 숫자와 그 뒤 문자를 분리
    num_str_pattern = re.compile(r"(\s\d+)([^\d\s])")
    sentence = re.sub(num_str_pattern, r"\1 \2", sentence)

    # 2. 공백으로 sentence를 분리 후 숫자인경우만 공백 넣어주기
    # numbers_reg = re.compile("\s\d{2,}\s")
    sentence_fixed = ""
    for token in sentence.split():
        if token.isnumeric():
            token = " ".join(token)
        sentence_fixed += " " + token
    return sentence_fixed


bracket_open_to_close = {
    "(": ")",
    "（": "）",
    "〈": "〉",
    "《": "》",
    "[": "]",
    "［": "］",
    "〔": "〕",
    "【": "】",
    "{": "}",
    "｛": "｝",
    "「": "」",
    "『": "』",
}
bracket_first_reg = "|".join([f"\{bracket_first}" for bracket_first in bracket_open_to_close.keys()])
bracket_last_reg = "|".join([f"\{bracket_last}" for bracket_last in bracket_open_to_close.values()])
bracket_all_reg = "".join(
    [f"\{bracket_first}\{bracket_last}" for bracket_first, bracket_last in bracket_open_to_close.items()]
)
bracket_rpat = re.compile(rf"({bracket_first_reg})(?!이하)[^{bracket_all_reg}]{{2,}}({bracket_last_reg})")

url_rpat1 = re.compile(r"https?://\S*|www\.\S*")
url_rpat2 = re.compile(r"[a-zA-Z0-9_]+(\.[a-zA-Z0-9_])+")
email_rpat = re.compile(r"[a-zA-Z0-9_]+@")
numbering_rpat = re.compile(r"(?:\[\d+\])|(?:[가나다라마바사아자차카타파하]\.\s)")  # 넘버링 [4] 가.  numbering_rpat
reporter_rpat = re.compile(r"\w+\s기자\s")
photo_rpat = re.compile(r"[<=/]*\s*사진\s*(설명|제공|출처){0,1}\s*[=>/]\s*[\w-]*\s*\b")

omit_rpat = re.compile(r"(\.){3,}|(-){2,}")  # 세 개 이상 점 또는 두개 이상 -은 ...와 같은 의미. 마스킹된 지워진 부분인듯
chinese_rpat = re.compile(r"\([一-龥]\)")  # 괄호안에 중국어만 있는거 삭제
dup_repat = re.compile(r"((-|\.)+)")  # ..는 . 로
raw_expression_rpat = re.compile(r"\b\w+(법|규칙|조례|칙|시행령)")


def pre_remove_noise(text):
    # text = text.lower()

    # 괄호 안에 조건부로 삭제
    def balanced(s):
        stack = []
        for c in s:
            if c in "".join(bracket_open_to_close.keys()):
                stack.append(c)
            elif stack and c == bracket_open_to_close[stack[-1]]:
                stack.pop()

        return len(stack) == 0

    if balanced(text):
        text = bracket_rpat.sub(" ", text)

    # url 대체
    text = url_rpat1.sub(r" ", text)
    text = url_rpat2.sub(r" ", text)
    text = email_rpat.sub(r" ", text)
    text = reporter_rpat.sub(r" ", text)
    text = photo_rpat.sub(r" ", text)

    # html 삭제
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 숫자 중간에 공백 삽입하기
    # text = number_split(text)
    # number_pattern = re.compile('\w*\d\w*')
    #     number_pattern = re.compile('\d+')
    #     text = number_pattern.sub(r'[[NUMBER]]', text)

    ######### unused token

    # unused_token으로 바꿀것들
    ## 다 동일하게 바꿔도되는거: unused10부터 시작

    # text = omit_rpat.sub("[unused10]", text)

    # print(text)

    # 중국어 처리
    # text = chinese_rpat.sub("", text)
    # for k, v in chinese2hangul_dict.items():
    #     text = text.replace(k, v)

    # 잘못 적힌거 정리
    text = dup_repat.sub(r"\2", text)

    # PUCTUACTION_TO_REMOVED = string.punctuation.translate(str.maketrans('', '', '\"\'#$%&\\@'))  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 중 적은것을 제외한 나머지를 삭제
    # text = text.translate(str.maketrans(PUCTUACTION_TO_REMOVED, ' '*len(PUCTUACTION_TO_REMOVED)))

    # remove_redundant_white_spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    # tgt special token 으로 활용할 204; 314[ 315] 대체/삭제해줘서 없애주기
    # text = re.sub("¶", " ", text)
    # text = re.sub("----------------", " ", text)
    # text = re.sub(";", ".", text)

    # result = []
    # morpher = mecab.MeCab()
    # for word in text.split():
    #     if raw_expression_rpat.search(word):
    #         result.append(" ".join(morpher.morphs(word)))
    #     else:
    #         result.append(word)
    # text = " ".join(result)

    # text = renaming_exp_rpat.sub(r" '\1'", text)
    # text = gwal_rpat.sub(r" ", text)

    # if tokenizer is not None:
    #     text = tokenizer(text)
    #     text = " ".join(text)

    return text


def post_remove_noise(text):
    text = url_rpat1.sub(r" ", text)
    text = url_rpat2.sub(r" ", text)
    text = email_rpat.sub(r" ", text)
    text = numbering_rpat.sub(r" ", text)
    text = reporter_rpat.sub(r" ", text)
    text = photo_rpat.sub(r" ", text)

    # html 삭제
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 잘못 적힌거 정리
    # text = dup_repat.sub(r"\2", text)
    text = text.replace('""', '"')
    text = text.replace("''", "'")
    # remove_redundant_white_spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def kr_sent_spliter(doc: Union[str, List], use_heuristic, num_workers):
    import kss

    kss_split_sent = functools.partial(
        kss.split_sentences,
        use_heuristic=use_heuristic,
        use_quotes_brackets_processing=False,
        max_recover_step=3,
        max_recover_length=20000,
        backend="mecab",
        num_workers=num_workers,
        disable_gc=True,
    )

    single_quate_not_pair_rpat = re.compile(r"(\"[^'\"]*)'([^'\"]*\")")
    double_quate_not_pair_rpat = re.compile(r"(\'[^'\"]*)\"([^'\"]*\')")

    def pre_cleaner(text):
        text = single_quate_not_pair_rpat.sub(r"\1 \2", text)
        text = double_quate_not_pair_rpat.sub(r"\1 \2", text)
        return text

    if isinstance(doc, list):
        doc = [pre_cleaner(text) for text in doc]
    else:
        doc = pre_cleaner(doc)

    return kss_split_sent(doc)
