# @Author : bamtercelboo
# @Datetime : 2018/8/16 8:50
# @File : Instance.py
# @Last Modify Time : 2018/8/16 8:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance.py
    FUNCTION : None
"""
import torch
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance(object):
    """
        Instance
    """
    def __init__(self):
        self.sentence = None
        self.sentence_id = None

