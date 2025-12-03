import os
from typing import List, Tuple

import jieba
import numpy as np
from tqdm import tqdm

from nlp.b_gen_vocabulary import text_preprocess, read_data_nmt

def text_split_en(text: str):
    """
    分词，这里采用空格切分，需要返回target和source两个list
    :param text: str 原始文
    :return:
    """
    source = []
    # 遍历每一行
    for i, line in enumerate(text.split('\n')):
        # 按照\t进行切分
        source.append(line.split(' '))
    return source


def text_split_cn(text: str):
    """
    分词，这里采用空格切分，需要返回target和source两个list
    :param text: str 原始文
    :return:
    """
    target = []
    # 遍历每一行
    for i, line in enumerate(text.split('\n')):
        target.append(list(jieba.cut(line)))  # 分词

    return target

class Vocabulary:
    PAD_TAG = "<pad>"  # 用PAD补全句子长度
    BOS_TAG = "<bos>"  # 用BOS表示开始
    EOS_TAG = "<eos>"  # 用EOS表示结束
    UNK_TAG = "<unk>"  # 用EOS表示结束
    PAD = 0  # PAD字符对应的数字
    BOS = 1  # BOS字符对应的数字
    EOS = 2  # EOS字符对应的数字
    UNK = 3  # UNK字符对应的数字

    def __init__(self):
        self.inverse_vocab = None
        self.vocabulary = {self.BOS_TAG: self.BOS, self.EOS_TAG: self.EOS,
                           self.PAD_TAG: self.PAD, self.UNK_TAG: self.UNK}
        self.count = {}  # 统计词频

    def fit(self, sentence_: List[str]):
        """
        统计词频
        """
        for word in sentence_:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=0, max=None, max_vocab_size=None) -> Tuple[dict, dict]:
        # 词频截断，词频大于或者小于一定数值时，舍弃
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        # 选择词表大小，根据词频排序后截断
        if max_vocab_size is not None:
            raw_len = len(self.count.items())
            vocab_size = max_vocab_size if raw_len > max_vocab_size else raw_len
            print('原始词表长度:{}，截断后长度:{}'.format(raw_len, vocab_size))
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:vocab_size]
            self.count = dict(temp)

        # 建立词表： token -> index
        for word in self.count:
            self.vocabulary[word] = len(self.vocabulary)
        # 词表翻转：index -> token
        self.inverse_vocab = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        return self.vocabulary, self.inverse_vocab

    def __len__(self):
        return len(self.vocabulary)


def get_vocab(text_list, path_out, max_vocab_size=3000):
    # 每个元素是list， 是一个切分好的句子
    vocab_hist = Vocabulary()
    for sentence in tqdm(text_list):
        vocab_hist.fit(sentence)
    vocab, inverse_vocab = vocab_hist.build_vocab(min=3, max_vocab_size=(max_vocab_size - 4))  # 3 是 pad\bos\eos\unk
    np.save(path_out, vocab)

    # 词表、词频可视化
    # print(len(vocab))
    # word_count = vocab_hist.count

    # out_dir = os.path.join(BASE_DIR, 'result')
    # path_w_frequency_out = os.path.join(out_dir, os.path.basename(path_out) + '_word_freq.jpg')
    # path_l_frequency_out = os.path.join(out_dir, os.path.basename(path_out) + '_length_freq.jpg')
    # plot_word_frequency(word_count, hist_size=100, path_out=path_w_frequency_out)
    # plot_sentence_length(text_list, hist_size=50, path_out=path_l_frequency_out)

data_dir = "./c2e"
path_train_en = os.path.join(data_dir, "t.en")
path_train_cn = os.path.join(data_dir, "t.cn")
path_test_cn = os.path.join(data_dir, "test.cn")
path_test_en = os.path.join(data_dir, "test.en")

text_train_e = read_data_nmt(path_train_en)
text_clean_train_e = text_preprocess(text_train_e)

text_train_c = read_data_nmt(path_train_cn)
text_clean_train_c = text_preprocess(text_train_c)

text_test_e = read_data_nmt(path_test_en)
text_clean_test_e = text_preprocess(text_test_e)

text_test_c = read_data_nmt(path_test_cn)
text_clean_test_c = text_preprocess(text_test_c)

train_split_cn = text_split_cn(text_clean_train_c)
train_split_en = text_split_en(text_clean_train_e)

out_dir = "./c2e"

vocab_path_en = os.path.join(out_dir, "vocab_en.npy")
vocab_path_cmn = os.path.join(out_dir, "vocab_cmn.npy")

get_vocab(train_split_en, vocab_path_en)
get_vocab(train_split_cn, vocab_path_cmn)


