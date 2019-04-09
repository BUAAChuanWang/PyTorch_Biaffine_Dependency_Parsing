# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:55
# @File : Batch_Iterator.py
# @Last Modify Time : 2018/1/30 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Batch_Iterator.py
    FUNCTION : None
"""

import torch
from torch.autograd import Variable
import random
import time
import numpy as np
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Batch_Features:
    """
    Batch_Features
    """
    def __init__(self):
        self.insts = None
        self.batch_size = 0
        self.words, self.ext_words, self.masks = 0, 0, 0
        self.heads, self.rels, self.lengths = 0, 0, 0

    @staticmethod
    def cuda(features, device):
        """
        :param features:
        :param device:
        :return:
        """
        features.words = features.words.to(device)
        features.ext_words = features.ext_words.to(device)
        features.masks = features.masks.to(device)


class Iterators:
    """
    Iterators
    """
    def __init__(self, batch_size=None, data=None, alphabet=None,
                 alphabet_ext=None, config=None):
        self.config = config
        self.device = config.device
        self.batch_size = batch_size
        self.data = data
        self.alphabet = alphabet
        self.alphabet_ext = alphabet_ext
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self):
        """
        :param batch_size:  batch size
        :param data:  data
        :param alphabet:
        :param config:
        :return:
        """
        start_time = time.time()
        assert isinstance(self.data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(self.batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        if len(self.data) != len(self.batch_size):
            print("Batch size Not Equal Data Count, Please Check.")
            exit()
        for id_data in range(len(self.data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self._convert_word2id(self.data[id_data], self.alphabet, self.alphabet_ext)
            self.features = self._Create_Each_Iterator(insts=self.data[id_data],
                                                       batch_size=self.batch_size[id_data])
            self.data_iter.append(self.features)
            self.features = []
        end_time = time.time()
        print("BatchIterator Time {:.4f}".format(end_time - start_time))
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def _convert_word2id(insts, alphabet, alphabet_ext):
        """
        :param insts:
        :param alphabet:
        :param alphabet_ext:
        :return:
        """
        for inst in insts:
            # copy with the word
            sent_id = []
            for dep in inst.sentence:
                # print(dep.head)
                id_dict = {"word": alphabet.word_alphabet.from_string(dep.form),
                           "ext_word": alphabet_ext.word_alphabet.from_string(dep.form),
                           "head": dep.head,
                           "rel": alphabet.rel_alphabet.from_string(dep.rel)}
                sent_id.append(id_dict)
            inst.sentence_id = sent_id
        print("Convert Finished.")

    def _Create_Each_Iterator(self, insts, batch_size):
        """
        :param insts:
        :param batch_size:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            # batch_flag = (inst.sentence_size != inst.next_sentence_size)
            if (len(batch) == batch_size) or (count_inst == len(insts)):
                one_batch = self._Create_Each_Batch(insts=batch)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, insts):
        """
        :param insts:
        :return:
        """
        alphabet, alphabet_ext = self.alphabet, self.alphabet_ext
        batch_size = len(insts)
        # copy with the max length for padding
        max_word_size = 0
        for inst in insts:
            word_size = len(inst.sentence)
            if word_size > max_word_size:
                max_word_size = word_size

        # create with the Tensor/Variable
        # word and label features
        # numpy array
        batch_word_features = np.full((batch_size, max_word_size), fill_value=alphabet.word_PADID, dtype=np.int64)
        batch_ext_word_features = np.full((batch_size, max_word_size), fill_value=alphabet_ext.word_PADID, dtype=np.int64)
        batch_mask_features = np.full((batch_size, max_word_size), fill_value=0, dtype=np.int64)
        batch_heads = []
        batch_rels = []
        batch_lengths = []

        for id_inst, inst in enumerate(insts):
            length = len(inst.sentence)
            batch_lengths.append(length)
            head = np.zeros((length), dtype=np.int32)
            rel = np.zeros((length), dtype=np.int32)
            for id_dep, dep in enumerate(inst.sentence):
                batch_word_features[id_inst, id_dep] = inst.sentence_id[id_dep]["word"]
                batch_ext_word_features[id_inst, id_dep] = inst.sentence_id[id_dep]["ext_word"]
                batch_mask_features[id_inst, id_dep] = 1
                head[id_dep] = inst.sentence_id[id_dep]["head"]
                rel[id_dep] = inst.sentence_id[id_dep]["rel"]
            batch_heads.append(head)
            batch_rels.append(rel)

        batch_word_features = torch.from_numpy(batch_word_features).long()
        batch_ext_word_features = torch.from_numpy(batch_ext_word_features).long()
        batch_mask_features = torch.from_numpy(batch_mask_features).float()

        # batch
        features = Batch_Features()
        features.insts = insts
        features.batch_size = batch_size
        features.words, features.ext_words, features.masks = batch_word_features, batch_ext_word_features, batch_mask_features
        features.heads, features.rels, features.lengths = batch_heads, batch_rels, batch_lengths

        if self.config.device != cpu_device:
            features.cuda(features, self.device)
        return features

    @staticmethod
    def _prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
        """
        :param inputs_words:
        :param seq_lengths:
        :param descending:
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long(), descending=descending)
        # sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_inputs_words = inputs_words[indices]
        return sorted_inputs_words, sorted_seq_lengths.cpu().numpy(), desorted_indices

