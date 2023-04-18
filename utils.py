# -*- coding: utf-8 -*-
"""
@Time ： 2022/2/23 15:40
@Auth ： Zhou Ru
"""

import os

import random
import logging
import torch
import numpy as np



def init_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # 使用FileHandler输出到文件
    num_list = []
    for filename in os.listdir('./log'):
        if 'log' not in filename:
            continue
        num = int(filename.split('.')[0][3:])
        num_list.append(num)
    num = max(num_list) + 1 if num_list != [] else 1
    fh = logging.FileHandler('./log/log{}.txt'.format(num))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger,num


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


import random

def random_weight(weight_data):
    total = sum(weight_data.values())  # 权重求和
    ra = random.uniform(0, total)  # 在0与权重和之前获取一个随机数
    curr_sum = 0
    ret = None
    keys = weight_data.keys()
    for k in keys:
        curr_sum += weight_data[k]  # 在遍历中，累加当前权重值
        if ra <= curr_sum:  # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret