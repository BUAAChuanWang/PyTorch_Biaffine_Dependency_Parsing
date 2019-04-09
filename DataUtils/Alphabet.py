# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:54
# @File : Alphabet.py
# @Last Modify Time : 2018/1/30 15:54
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Alphabet.py
    FUNCTION : None
"""

import os
import sys
import time
import torch
import random
import collections
from collections import Counter
from DataUtils.Common import seed_num, UNK, PAD
torch.manual_seed(seed_num)
random.seed(seed_num)


class CreateAlphabet:
    """
        Class:      Create_Alphabet
        Function:   Build Alphabet By Alphabet Class
        Notice:     The Class Need To Change So That Complete All Kinds Of Tasks
    """
    def __init__(self, min_freq=1, train_data=None, dev_data=None, test_data=None,
                 config=None):
        """
        :param min_freq:
        :param train_data:
        :param dev_data:
        :param test_data:
        :param config:
        """

        # time
        self.start_time = 0
        self.end_time = 0

        # minimum vocab size
        self.min_freq = min_freq

        self.config = config

        # Data
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.datasets = self._build_data(train_data=train_data, dev_data=dev_data, test_data=test_data)

        self.word_counter, self.rel_counter, self.relroot = self._counter(self.datasets)
        self._root = self.relroot
        self._root_form = '<' + self.relroot.lower() + '>'

        # storage word and label
        self.word_state = collections.OrderedDict()
        self.rel_state = collections.OrderedDict()

        ### word --- UNK PAD ROOT
        self.word_state[PAD] = self.min_freq
        self.word_state[UNK] = self.min_freq
        self.word_state[self._root_form] = self.min_freq

        ### rel --- PAD ROOT
        self.rel_state[PAD] = self.min_freq
        self.rel_state[self.relroot] = self.min_freq

        # word and label Alphabet
        self.word_alphabet = Alphabet(min_freq=self.min_freq)
        self.rel_alphabet = Alphabet()

        ### word --- UNK PAD ROOT key
        self.word_PADID = -1
        self.word_UNKID = -1
        self.word_ROOTID = -1

        # rel --- PAD ROOT key
        self.rel_PADID = -1
        self.rel_ROOTID = -1

    @staticmethod
    def _build_data(train_data=None, dev_data=None, test_data=None):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        """
        # handle the data whether to fine_tune
        """
        :param train data:
        :param dev data:
        :param test data:
        :return: merged data
        """
        assert train_data is not None, "The Train Data Is Not Allow Empty."
        datasets = []
        datasets.extend(train_data)
        print("the length of train data {}".format(len(datasets)))
        if dev_data is not None:
            print("the length of dev data {}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("the length of test data {}".format(len(test_data)))
            datasets.extend(test_data)
        print("the length of data that create Alphabet {}".format(len(datasets)))
        return datasets

    @staticmethod
    def _counter(data):
        word_counter = Counter()
        rel_counter = Counter()
        root = ''
        for inst in data:
            for dep in inst.sentence:
                word_counter[dep.form] += 1
                if dep.head != 0:
                    rel_counter[dep.rel] += 1
                elif root == '':
                    root = dep.rel
                    rel_counter[dep.rel] += 1
                elif root != dep.rel:
                    print('root = ' + root + ', rel for root = ' + dep.rel)
        return word_counter, rel_counter, root

    def build_vocab(self):
        """
        :return:
        """
        print("Build Vocab Start...... ")
        self.start_time = time.time()

        # create the word Alphabet
        for word, count in self.word_counter.most_common():
            self.word_state[word] = count

        for rel, count in self.rel_counter.most_common():
            if rel != self.relroot:
                self.rel_state[rel] = count

        # Create id2words and words2id by the Alphabet Class
        self.word_alphabet.initial(self.word_state)
        self.rel_alphabet.initial(self.rel_state)

        ### word --- UNK PAD ROOT key
        self.word_UNKID = self.word_alphabet.from_string(UNK)
        self.word_PADID = self.word_alphabet.from_string(PAD)
        self.word_ROOTID = self.word_alphabet.from_string(self._root_form)

        # rel --- PAD ROOT key
        self.rel_PADID = self.rel_alphabet.from_string(PAD)
        self.rel_ROOTID = self.rel_alphabet.from_string(self.relroot)

        # fix the vocab
        self.word_alphabet.set_fixed_flag(True)
        self.rel_alphabet.set_fixed_flag(True)

        self.end_time = time.time()
        print("Build Vocab Finished.")
        print("Build Vocab Time is {:.4f}.".format(self.end_time - self.start_time))


class Alphabet:
    """
        Class: Alphabet
        Function: Build vocab
        Params:
              ******    id2words:   type(list),
              ******    word2id:    type(dict)
              ******    vocab_size: vocab size
              ******    min_freq:   vocab minimum freq
              ******    fixed_vocab: fix the vocab after build vocab
              ******    max_cap: max vocab size
    """
    def __init__(self, min_freq=1):
        self.id2words = []
        self.words2id = collections.OrderedDict()
        self.vocab_size = 0
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def initial(self, data):
        """
        :param data:
        :return:
        """
        for key in data:
            if data[key] >= self.min_freq:
                self.from_string(key)
        self.set_fixed_flag(True)

    def set_fixed_flag(self, bfixed):
        """
        :param bfixed:
        :return:
        """
        self.fixed_vocab = bfixed
        if (not self.fixed_vocab) and (self.vocab_size >= self.max_cap):
            self.fixed_vocab = True

    def from_string(self, string):
        """
        :param string:
        :return:
        """
        if string in self.words2id:
            return self.words2id[string]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(string)
                self.words2id[string] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                # return -1
                return self.words2id[UNK]

    def from_id(self, qid, defineStr=""):
        """
        :param qid:
        :param defineStr:
        :return:
        """
        if int(qid) < 0 or self.vocab_size <= qid:
            return defineStr
        else:
            return self.id2words[qid]

    def initial_from_pretrain(self, pretrain_file, unk, padding):
        """
        :param pretrain_file:
        :param unk:
        :param padding:
        :return:
        """
        print("initial alphabet from {}".format(pretrain_file))
        self.from_string(unk)
        self.from_string(padding)
        now_line = 0
        with open(pretrain_file, encoding="UTF-8") as f:
            for line in f.readlines():
                now_line += 1
                sys.stdout.write("\rhandling with {} line".format(now_line))
                info = line.split(" ")
                self.from_string(info[0])
        f.close()
        print("\nHandle Finished.")



