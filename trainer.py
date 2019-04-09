# @Author : bamtercelboo
# @Datetime : 2018/8/26 8:30
# @File : trainer.py
# @Last Modify Time : 2018/8/26 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  trainer.py
    FUNCTION : None
"""

import os
import sys
import time
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from Dataloader.DataLoader import batch_variable_depTree
from Dataloader.Dependency import evalDepTree
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        Args of data:
            train_iter : train batch data iterator
            dev_iter : dev batch data iterator
            test_iter : test batch data iterator
        Args of train:
            model : nn model
            config : config
        """
        print("Training Start......")
        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.parser = kwargs["model"]
        self.config = kwargs["config"]
        self.device = self.config.device
        self.cuda = False
        if self.device != cpu_device:
            self.cuda = True
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.parser.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        if self.config.learning_algorithm == "SGD":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        else:
            self.loss_function = nn.CrossEntropyLoss(reduction="mean")
        print(self.optimizer)
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.parser.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            # print("epoch", epoch)
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):
        """
        Args:
            epoch: int, epoch
            init_lr: initial lr
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        # print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :return:
        """
        if backward_count % config.update_batch_size == 0 or backward_count == self.train_iter_len:
            self._clip_model_norm(self.config.clip_max_norm_use, self.config.clip_max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch):
        """
        :param epoch:
        :return:
        """
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            print("Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                print("Early Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                exit()

    def _model2file(self, model, config, epoch):
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def train(self):
        """
        :return:
        """
        epochs = self.config.epochs
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            new_lr = self._dynamic_lr(config=self.config, epoch=epoch, new_lr=new_lr)
            # self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now lr is {}".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.parser.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                words, ext_words, masks = batch_features.words, batch_features.ext_words, batch_features.masks
                heads, rels, lengths = batch_features.heads, batch_features.rels, batch_features.lengths
                sumLength = sum(lengths)
                self.parser.forward(words, ext_words, masks)

                loss = self.parser.compute_loss(heads, rels, lengths)
                loss = loss / self.config.update_batch_size
                loss_value = loss.data.cpu().numpy()
                loss.backward()

                self._optimizer_batch_step(config=self.config, backward_count=backward_count)

                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    arc_correct, label_correct, total_arcs = self.parser.compute_accuracy(heads, rels)
                    overall_arc_correct += arc_correct
                    overall_label_correct += label_correct
                    overall_total_arcs += total_arcs
                    uas = overall_arc_correct.item() * 100.0 / overall_total_arcs
                    las = overall_label_correct.item() * 100.0 / overall_total_arcs
                    sys.stdout.write(
                        "\nbatch_count = [{}/{}] , loss is {:.6f}, length: {}, ARC: {:.6f}, REL: {:.6f}".format(
                            batch_count + 1, self.train_iter_len, float(loss_value), sumLength, float(uas), float(las)))
            end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(parser=self.parser, epoch=epoch, config=self.config)
            self._model2file(model=self.parser.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch)

    def eval(self, parser, epoch, config):
        """
        :param parser:
        :param epoch:
        :param config:
        :return:
        """
        eval_start_time = time.time()
        self._eval_batch(self.dev_iter, parser, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        eval_start_time = time.time()
        self._eval_batch(self.test_iter, parser, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    # self.get_one_batch(batch_features.insts)
    def get_one_batch(self, insts):
        """
        :param insts:
        :return:
        """
        batch = []
        for inst in insts:
            batch.append(inst.sentence)
        return batch

    def _eval_batch(self, data_iter, parser, best_score, epoch, config, test=False):
        """
        :param data_iter:
        :param parser:
        :param vocab:
        :param best_score:
        :param epoch:
        :param config:
        :param test:
        :return:
        """
        parser.model.eval()
        arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
        alphabet = config.alphabet

        for batch_count, batch_features in enumerate(data_iter):
            one_batch = self.get_one_batch(batch_features.insts)
            words, ext_words, masks = batch_features.words, batch_features.ext_words, batch_features.masks
            heads, rels, lengths = batch_features.heads, batch_features.rels, batch_features.lengths
            sumLength = sum(lengths)
            count = 0
            arcs_batch, rels_batch = parser.parse(words, ext_words, lengths, masks)
            for tree in batch_variable_depTree(one_batch, arcs_batch, rels_batch, lengths, alphabet):
                # printDepTree(output, tree)
                arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, one_batch[count])
                arc_total_test += arc_total
                arc_correct_test += arc_correct
                rel_total_test += rel_total
                rel_correct_test += rel_correct
                count += 1

        uas = arc_correct_test * 100.0 / arc_total_test
        las = rel_correct_test * 100.0 / rel_total_test

        f = uas
        # p, r, f = law_p, law_r, law_f

        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = f
            if f >= best_score.best_dev_score:
                best_score.best_dev_score = f
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.f = f
        print("{}:".format(test_flag))
        print("UAS = %d/%d = %.2f, LAS = %d/%d =%.2f" % (arc_correct_test, arc_total_test, uas,
                                                         rel_correct_test, rel_total_test, las))

        if test is True:
            print("The Current Best Dev score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score, best_score.best_epoch))
        if test is True:
            best_score.best_test = False





