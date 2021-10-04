import gc
import glob
import hashlib
import itertools
import json
import os
from os.path import join as pjoin
import sys
import random
import re
import subprocess
from tqdm import tqdm

import torch
from multiprocess import Pool
import xml.etree.ElementTree as ET

from itertools import permutations
import numpy as np
import pandas as pd

# import mecab

# from others.tokenization import BertTokenizer
from transformers import BertTokenizerFast, ElectraTokenizerFast
from prepro.tokenization_kobert import KoBertTokenizer

from prepro.preprocessor_kr import pre_remove_noise
from prepro.utils import _get_word_ngrams, make_or_initial_dir, get_subdata_group, load_obj, save_obj
from others.utils import clean
from others.rouge_metric import Rouge
from others.logging import logger
import base

tqdm.pandas()
# morpher = mecab.MeCab()

# unused_tok_rpat = re.compile(r"\[\s*unused\s*(\d+)\s*\]")

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r" \'{\w}", "'\g<1>", s)
    s = re.sub(r"\'\' {\w}", "''\g<1>", s)


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))["sentences"]:
        tokens = [t["word"] for t in sent["tokens"]]
        if lower:
            tokens = [t.lower() for t in tokens]
        if tokens[0] == "@highlight":
            flag = True
            tgt.append([])
            continue
        if flag:
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(" ".join(sent)).split() for sent in source]
    tgt = [clean(" ".join(sent)).split() for sent in tgt]
    return source, tgt


def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter("hedline"))
    if len(title_node) > 0:
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter("hl1"))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter("byline"))
    byline_node = [n for n in byline_node if n.attrib["class"] == "normalized_byline"]
    if len(byline_node) > 0:
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter("abstract"))
    if len(abs_node) > 0:
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter("p"))][0]
        except:
            print(p)

    else:
        return None, None
    abs = " ".join(abs).split(";")
    abs[-1] = abs[-1].replace("(m)", "")
    abs[-1] = abs[-1].replace("(s)", "")

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace("(" + ww + ")", "")
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter("block"):
        att = doc_node.get("class")
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if att == "full_text":
            paras = [p.text.lower().split() for p in list(doc_node.iter("p"))]
            break
    if len(paras) > 0:
        if len(byline) > 0:
            paras = [title + ["[unused3]"] + byline + ["[unused4]"]] + paras
        else:
            paras = [title + ["[unused3]"]] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if not s.endswith("story"):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = [
        "java",
        "edu.stanford.nlp.pipeline.StanfordCoreNLP",
        "-annotators",
        "tokenize,ssplit",
        "-ssplit.newlineIsSentenceBreak",
        "always",
        "-filelist",
        "mapping_for_corenlp.txt",
        "-outputFormat",
        "json",
        "-outputDirectory",
        tokenized_stories_dir,
    ]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?"
            % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig)
        )
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


# 1개만 선택했을 때 가장 높은거 고르고, 그 1개와 다른 1개 조합했을 때 가장 높은거 고르고... 차례대로
# 만약 1개만 선택한것보다 다른 1개 더 선택하는 조합의 루지 점수가 떨어지면 그냥 1개만 나올 수 도 있음
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9가-힣 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return selected  # sorted(selected)


# 전체 경우의 수 탐색
def full_selection(doc_sent_list, abstract_sent_list, summary_size=3):
    def _rouge_clean(s):
        return re.sub(r"[^A-Za-z0-9가-힣 ]", "", s)

    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
    )

    # cleaning and merge [[w,w,w], [w,w,w]] -> [w,w,w, w,w,w]
    abstract = sum(abstract_sent_list, [])  # [[w,w,w], [w,w,w]] -> [w,w,w, w,w,w]
    abstract = _rouge_clean(" ".join(abstract))
    doc_sent_list_merged = [_rouge_clean(" ".join(sent)) for sent in doc_sent_list]
    src_len = len(doc_sent_list_merged)

    # 일단 greedy로 구한 다음 3개가 안되는 경우만 나머지를 full로 채움!
    selected_idx3_list = greedy_selection(doc_sent_list, abstract_sent_list, summary_size)
    # print('1 ', selected_idx3_list)
    # if len(selected_idx3_list) == 3:
    #     return selected_idx3_list

    total_max_rouge_score = 0.0
    if src_len > 10 or (src_len <= 10 and len(selected_idx3_list) < 2):  # greedy
        for i in range(summary_size - len(selected_idx3_list)):

            # cur_sents_idx3_list = []
            cur_max_total_rouge_score = 0.0
            cur_sent_idx = -1
            for sent_idx, sent in enumerate(doc_sent_list_merged):

                if sent_idx in selected_idx3_list:
                    continue
                temp_idx3_list = selected_idx3_list + [sent_idx]
                sents_array = np.array(doc_sent_list_merged)[temp_idx3_list]
                sents_merged = " ".join(sents_array)
                # print('temp_idx3_list', temp_idx3_list)
                # ROUGE1,2,l 합score 계산
                rouge_scores = rouge_evaluator.get_scores(sents_merged, abstract)
                total_rouge_score = 0
                for k, v in rouge_scores.items():
                    total_rouge_score += v["f"]
                # print('total_rouge_score', total_rouge_score)
                if total_rouge_score > cur_max_total_rouge_score:
                    cur_max_total_rouge_score = total_rouge_score
                    cur_sent_idx = sent_idx
                #  print(cur_max_total_rouge_score)
                # print(selected_idx3_list)
            selected_idx3_list.append(cur_sent_idx)
            # print('-----------------------')
            total_max_rouge_score = cur_max_total_rouge_score
    # print('2 ', selected_idx3_list)

    # full
    sents_idx_perm_list = list(permutations(range(src_len), summary_size))
    sents_idx_list = []
    for sents_idx_perm in sents_idx_perm_list:
        if set(sents_idx_perm) & set(selected_idx3_list) == set(selected_idx3_list):
            sents_idx_list.append(sents_idx_perm)

    for sents_idx in sents_idx_list:
        sents_array = np.array(doc_sent_list_merged)[list(sents_idx)]
        sents_merged = " ".join(sents_array)
        # print(sents_merged)
        # print(sents_idx)
        # print(sents_array)

        # ROUGE1,2,l 합score 계산
        rouge_scores = rouge_evaluator.get_scores(sents_merged, abstract)
        total_rouge_score = 0
        for k, v in rouge_scores.items():
            total_rouge_score += v["f"]

        if total_rouge_score > total_max_rouge_score:
            total_max_rouge_score = total_rouge_score
            selected_idx3_list = list(sents_idx)
    # print('3 ', selected_idx3_list)
    return selected_idx3_list  # , total_max_rouge_score,  sorted(selected_idx3_list)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


class BertData:
    used_subtoken_idxs = set()

    def __init__(self, args):
        self.args = args

        if args.model_name == "monologg/kobert":
            self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert", do_lower_case=True)
            self.sep_token = "[SEP]"
            self.cls_token = "[CLS]"
            self.pad_token = "[PAD]"
            self.tgt_bos = "¶"  # '[unused0]'   204; 314[ 315]
            self.tgt_eos = "----------------"  # '[unused1]'
            self.tgt_sent_split = ";"  #'[unused2]'

            self.sep_vid = self.tokenizer.token2idx[self.sep_token]
            self.cls_vid = self.tokenizer.token2idx[self.cls_token]
            self.pad_vid = self.tokenizer.token2idx[self.pad_token]

        elif args.model_name in [
            "kykim/bert-kor-base",
            "kykim/albert-kor-base",
            "kykim/electra-kor-base",
        ]:
            if args.model_name == "kykim/electra-kor-base":
                self.tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)
            else:
                self.tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)
            self.sep_token = "[SEP]"
            self.cls_token = "[CLS]"
            self.pad_token = "[PAD]"
            self.tgt_bos = "[unused0]"  # '[unused0]'   204; 314[ 315]
            self.tgt_eos = "[unused1]"  # '[unused1]'
            self.tgt_sent_split = "[unused2]"  #'[unused2]'

            self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
            self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
            self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):
        """
        ext에서는 sent_labels만 사용되고, tgt_subtoken_idxs는 만들어지긴 하는데 결국 사용되지는 않음.
        """

        if (not is_test) and len(src) == 0:
            return None

        original_src_txt = [" ".join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1
        # print(sent_labels)
        # print(_sent_labels)
        src = [src[i][: self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        # print(idxs)
        # print(sent_labels)
        src = src[: self.args.max_src_nsents]
        sent_labels = sent_labels[: self.args.max_src_nsents]

        if (not is_test) and len(src) < self.args.min_src_nsents:
            return None

        src_txt = [" ".join(sent) for sent in src]
        text = " {} {} ".format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[: len(cls_ids)]

        # kobert transforemrs에 연결되어 있는 transforemrs tokenizer 사용
        tgt_subtokens_str = (
            self.tgt_bos
            + " "
            + f" {self.tgt_sent_split} ".join([" ".join(self.tokenizer.tokenize(" ".join(tt))) for tt in tgt])
            + " "
            + self.tgt_eos
        )
        ## presumm tokenizer 사용
        # """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
        # tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
        #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'

        tgt_subtoken = tgt_subtokens_str.split()[: self.args.max_tgt_ntokens]
        if (not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens:
            return None
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        # print(tgt_subtoken)
        # print(tgt_subtoken_idxs)
        tgt_txt = "<q>".join([" ".join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # BertData.used_subtoken_idxs.update(src_subtoken_idxs)
        # BertData.used_subtoken_idxs.update(tgt_subtoken_idxs)
        # print(sent_labels)
        return (
            src_subtoken_idxs,
            sent_labels,
            tgt_subtoken_idxs,
            segments_ids,
            cls_ids,
            src_txt,
            tgt_txt,
        )


def format_to_bert(args):
    if args.dataset != "":
        datasets = [args.dataset]
    else:
        datasets = ["train", "valid", "test"]
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, "*" + corpus_type + ".*.json")):
            real_name = json_f.split("/")[-1]
            a_lst.append(
                (
                    corpus_type,
                    json_f,
                    args,
                    pjoin(args.save_path, real_name.replace("json", "bert.pt")),
                )
            )
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == "test"
    if os.path.exists(save_file):
        logger.info("Ignore %s" % save_file)
        return

    bert = BertData(args)
    logger.info("Processing %s" % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d["src"], d["tgt"]
        # print(source)

        if "idx" in args.tgt_type:
            sent_labels = tgt  # index
            tgt = [source[idx] for idx in sent_labels]  # []  list로 아래서 가정해야 하기에..
            # print("ddd", sent_labels)
        else:
            sent_labels = full_selection(source[: args.max_src_nsents], tgt, 3)
            # sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        # print(sent_labels)

        if args.lower:
            source = [" ".join(s).lower().split() for s in source]
            tgt = [" ".join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(
            source,
            tgt,
            sent_labels,
            use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
            is_test=is_test,
        )
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if b_data is None:
            continue
        (
            src_subtoken_idxs,
            sent_labels,
            tgt_subtoken_idxs,
            segments_ids,
            cls_ids,
            src_txt,
            tgt_txt,
        ) = b_data
        b_data_dict = {
            "src": src_subtoken_idxs,
            "tgt": tgt_subtoken_idxs,
            "src_sent_labels": sent_labels,
            "segs": segments_ids,
            "clss": cls_ids,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }  ##  (원복)원래 키값이 src_txt, tgt_txt 이었는데 수정!!!!!
        # print(b_data_dict)
        datasets.append(b_data_dict)
    logger.info("Processed instances %d" % len(datasets))

    # with open(args.save_path + '/used_subtoken_idxs.txt', 'a') as f:
    #     for idx in BertData.used_subtoken_idxs:
    #         f.write("%s\n" % idx)

    logger.info("Saving to %s" % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ["valid", "test", "train"]:
        temp = []
        for line in open(pjoin(args.map_path, "mapping_" + corpus_type + ".txt")):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, "*.json")):
        real_name = f.split("/")[-1].split(".")[0]
        if real_name in corpus_mapping["valid"]:
            valid_files.append(f)
        elif real_name in corpus_mapping["test"]:
            test_files.append(f)
        elif real_name in corpus_mapping["train"]:
            train_files.append(f)
        # else:
        #     train_files.append(f)

    corpora = {"train": train_files, "valid": valid_files, "test": test_files}
    for corpus_type in ["train", "valid", "test"]:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, "w") as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if len(dataset) > 0:
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, "w") as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {"src": source, "tgt": tgt}


def format_xsum_to_lines(args):
    if args.dataset != "":
        datasets = [args.dataset]
    else:
        datasets = ["train", "test", "valid"]

    corpus_mapping = json.load(open(pjoin(args.raw_path, "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, "restbody")
        root_tgt = pjoin(args.raw_path, "firstsentence")
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if d is None:
                continue
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, "w") as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if len(dataset) > 0:
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, "w") as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + ".restbody")
    f_tgt = pjoin(root_tgt, name + ".fs")
    if os.path.exists(f_src) and os.path.exists(f_tgt):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {"src": source, "tgt": tgt}
    return None


def format_df_to_json(df, src_name, tgt_name, tgt_type, to_dir, subdata_group):
    NUM_DOCS_IN_ONE_FILE = 1000
    start_idx_list = list(range(0, len(df), NUM_DOCS_IN_ONE_FILE))

    for start_idx in tqdm(start_idx_list):
        end_idx = start_idx + NUM_DOCS_IN_ONE_FILE
        if end_idx > len(df):
            end_idx = len(df)  # -1로 하니 안됨...

        # 정렬을 위해 앞에 0 채워주기
        length = len(str(len(df)))
        start_idx_str = (length - len(str(start_idx))) * "0" + str(start_idx)
        end_idx_str = (length - len(str(end_idx - 1))) * "0" + str(end_idx - 1)

        file_name = f"{to_dir}/{subdata_group}.{start_idx_str}_{end_idx_str}.json"

        json_list = []
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            original_sents_list = [sent.split() for sent in row[src_name]]

            summary_sents_list = []
            if subdata_group in ["train", "valid"]:
                summary_sents_list = row[tgt_name]
                if "idx" in tgt_type:
                    summary_sents_list = row[
                        tgt_name
                    ]  # [sent.split() for sent in row[src_name][row[tgt_name]]] # bert로 변환할 때 텍스트 변환해줌. extract할때 idx 도 써야하기에..
                else:
                    summary_sents_list = [sent.split() for sent in row[tgt_name]]

                # print("aaa", tgt_name, summary_sents_list)
            json_list.append({"src": original_sents_list, "tgt": summary_sents_list})

        json_string = json.dumps(json_list, indent=4, ensure_ascii=False)
        # print(json_string)
        with open(file_name, "w") as json_file:
            json_file.write(json_string)


def format_df_to_bert(cfg):
    text, summary = cfg.dataset.text, cfg.dataset.summary
    # 실행을 다른 곳에서 해도 일단 df 경로는 찾을 수 있도록... 수정해주기
    # dataset_dir = os.path.join(base.BASE_DIR, "datasets", cfg.dataset.name)
    # dirs = {k: os.path.join(dataset_dir, v) for k, v in cfg.dirs.items()}
    dirs = cfg.dirs

    def _format_df_to_bert(df, subdata_group):
        logger.info(f"Cleaning...")
        if text.do_cleaning:
            if "list" in text.type:
                df[text.name] = df[text.name].progress_apply(
                    lambda text_list: [pre_remove_noise(text) for text in text_list]
                )
            else:
                df[text.name] = df[text.name].progress_apply(pre_remove_noise)

            if subdata_group in ["train", "valid"] and "idx" not in summary.type:
                text_len_arr = (
                    df[text.name].apply(lambda text_list: len(" ".join(text_list)))
                    if "list" in text.type
                    else df[text.name].str.len()
                )
                summary_len_arr = (
                    df[summary.name].apply(lambda summary_list: len(" ".join(summary_list)))
                    if "list" in summary.type
                    else df[summary.name].str.len()
                )
                df = df[text_len_arr > summary_len_arr]

        logger.info(f"Sent split...")
        if "list" not in text.type:
            from prepro.preprocessor_kr import kr_sent_spliter

            df[text.name] = pd.Series(
                kr_sent_spliter(
                    list(df[text.name].values), use_heuristic=cfg.dataset.is_informal, num_workers=cfg.n_cpus
                )
            )

        if subdata_group in ["train", "valid"] and "list" not in summary.type:
            from prepro.preprocessor_kr import kr_sent_spliter

            df[summary.name] = pd.Series(
                kr_sent_spliter(
                    list(df[summary.name].values), use_heuristic=cfg.dataset.is_informal, num_workers=cfg.n_cpus
                )
            )

        # df to json file
        format_df_to_json(df, text.name, summary.name, summary.type, dirs.json, subdata_group=subdata_group)

        # json to bert.pt files
        # base_path = os.getcwd()
        make_or_initial_dir(os.path.join(dirs.bert, subdata_group))
        os.system(
            f"python {os.path.join(base.BASE_DIR, 'src', 'preprocess.py')}"
            + f" -mode format_to_bert -dataset {subdata_group}"
            + f" -model_name {cfg.model_name }"
            + f" -tgt_type {summary.type}"
            + f" -raw_path {dirs.json}"
            + f" -save_path {dirs.bert}"
            # + f" -log_file {dirs.json}"
            + f" -lower -n_cpus {cfg.n_cpus}"
        )

    file_paths = glob.glob(os.path.join(dirs["df"], "*"))  # "pkl"
    print(file_paths)
    if not file_paths:
        logger.error(f"There is no 'df' files in {dirs.df}")
        sys.exit("Stop")

    # 동일한 파일명 존재하면 덮어쓰는게 아니라 ignore됨에 따라 폴더 내 삭제 후 만들어주기
    make_or_initial_dir(dirs.json)
    make_or_initial_dir(dirs.bert)
    # os.makedirs(to_dir, exist_ok=True)

    for df_file in file_paths:
        logger.info(f"Start 'df_to_bert' processing for {df_file}")
        df = pd.read_pickle(df_file)
        # print(df)
        subdata_group = get_subdata_group(df_file)

        _format_df_to_bert(df, subdata_group)
