import os
import pickle
import json
import glob
from typing import Collection, List, Union
from varname import argname
from others.logging import logger

# stopwords = pkgutil.get_data(__package__, 'smart_common_words.txt')
# stopwords = stopwords.decode('ascii').split('\n')
# stopwords = {key.strip(): 1 for key in stopwords}


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def make_or_initial_dir(dir_path):
    """
    Make dir(When dir exists, remove it and make it)
    """
    if os.path.exists(dir_path):
        os.system(f"rm -rf {dir_path}/")
        logger.info(f"{dir_path} folder is removed")

    os.mkdir(dir_path)
    logger.info(f"{dir_path} folder is made")


def get_subdata_group(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    types = ["train", "valid", "test"]
    for type in types:
        if type in filename:
            return type


def load_obj(path: str, file_type: str = None):
    """[summary]
    추가 개발할 것
    - 로드된 obj에 종류를 알고, 여러개일 떄 하나로 합쳐서 반환하기
    - 멀티프로세싱 적용
    """
    if os.path.isdir(path):
        file_paths = glob.glob(path)

        if not file_paths:
            raise FileNotFoundError

        if not file_type:
            # Check whether the files has same file_type
            file_extension_set = set()
            for file_path in file_paths:
                _, file_extension = os.path.splitext(os.path.basename(file_path))
                file_extension_set.add(file_extension)
            assert len(file_extension_set) == 1

            file_type = file_extension_set.pop()

        obj_list = []
        for file_path in file_paths:
            obj_list.append(_load_obj(file_path, file_type))

        print(f"Loaded {len(file_paths)} {file_type} files from {path}")
        return obj_list

    elif os.path.isfile(path):
        _, file_extension = os.path.splitext(os.path.basename(path))
        obj = _load_obj(path, file_extension)
        print(f"Loaded from {path}")

        return obj


def _load_obj(path: str, file_type: str):
    if file_type[0] != ".":
        file_type = "." + file_type

    with open(path, "rb") as f:  # , encoding="utf-8-sig"
        if file_type in [".pickle", ".pkl", "p"]:
            return pickle.load(f)

        elif file_type == ".json":
            return json.load(f)

        elif file_type == ".jsonl":
            json_list = list(f)
            jsons = []
            for json_str in json_list:
                line = json.loads(json_str)  # 문자열을 읽을때는 loads
                jsons.append(line)
            return jsons

        if file_type == ".plain":
            lines = f.read().splitlines()
            return lines


def save_obj(collection: Collection, path: str):
    # If path designate dir, make it as file path(file name as collection var name)
    _, file_extension = os.path.splitext(os.path.basename(path))
    if file_extension == "":
        path = f"{path}/{argname(collection)}.pkl"
        file_extension = ".pkl"

    # Make dirs
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)

    if file_extension in ["pickle", "pkl", "p"]:
        with open(path, "wb") as f:
            pickle.dump(collection, f)

    else:
        with open(path, "w") as f:
            if file_extension == "jsonl":
                for item in collection:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            else:  # txt ...
                f.write("\n".join(collection))

    print(f"Save to {path}")
