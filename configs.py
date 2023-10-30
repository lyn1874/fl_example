#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/03/05 16:14:14
@Author  :   Bo 
'''

import sys
import argparse 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='VAE-Reconstruction')
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--split", type=str, default="by_cls")
    parser.add_argument("--shuffle_percentage", type=float, default=0.0)
    parser.add_argument("--seed_use", type=int, default=1024)
    parser.add_argument("--num_local_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--method", type=str, default="check_zeta")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="m_mlp")
    parser.add_argument("--initial_seed_use", type=int, default=1024)
    parser.add_argument("--use_local_id", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--non_iid_alpha", type=float, default=0.1)
    parser.add_argument("--partition_type", type=str, default="non_iid")
    parser.add_argument("--dir_name", type=str)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--opt", type=str, default="client")
    

    return parser.parse_args()
