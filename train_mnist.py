#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/03/05 16:14:14
@Author  :   Bo 
'''
import mnist_utils as mnist_utils 
import numpy as np 
import torch 
import os 
import torch.nn as nn 
from tqdm import tqdm 
import sys
import argparse 
import time 
import scipy 
import math 
import copy 

device=torch.device("cuda")

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
    return parser.parse_args()


def get_model_grads(model):
    return [p.grad.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_params(model):
    return [p.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]


class Train(object):
    def __init__(self, conf, data_group, num_local_epochs, sigma, exist_model, version=0):
        self.conf = conf 
        self.data_group = data_group 
        self.num_local_epochs = num_local_epochs
        self.sigma = sigma 
        self.model_use = mnist_utils.CLSModel(28*28, 10).to(device)        
        
        if conf.round > 0:
            self.model_use.load_state_dict(exist_model)
        
        self.optimizer = mnist_utils.define_optimizer(self.model_use, lr=conf.lr)                
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        self.tr_data_loader, self.tt_data_loader = data_group
        self.version = version 
        
        print("Training data size", len(self.tr_data_loader))
        print("Testing data size", len(self.tt_data_loader))
        
        parameter_list = [p for p in self.model_use.parameters() if p.requires_grad == False]
        assert len(parameter_list) == 0 
         
        
    def get_grad(self):
        grad_group = []
        for name, p in self.model_use.named_parameters():
            if p.requires_grad and "bias" not in name:
                grad_group.append(np.reshape(p.grad.data.detach().cpu().numpy(), [-1]))
        return grad_group[0]
                        
    def _update_batch_tr(self, _image, _label, global_step):
        _image, _label = _image.to(device), _label.to(device)
        prev_model = copy.deepcopy(self.model_use)
        self.optimizer.zero_grad()
        _pred = self.model_use(_image)
        _loss = self.loss_fn(_pred, _label) / len(_image)
        _loss.backward()
        
        self.optimizer.step()
        
        accu = (_pred.argmax(axis=-1) == _label).sum().div(len(_image))
        print("Training loss: {:.4f} and Training accuracy {:.2f}".format(_loss.item(), accu.item()))
        return self.get_grad()
        
    def _eval(self, global_step, data_use, str_use):
        self.model_use.eval()
        val_loss, val_accu = 0.0, 0.0
        for i, (_image, _label) in enumerate(data_use):
            _image, _label = _image.to(device), _label.to(device)

            _pred = self.model_use(_image)
            _loss = self.loss_fn(_pred, _label) 
            _accu = (_pred.argmax(axis=-1) == _label).sum()
            val_loss += _loss.detach().cpu().numpy()
            val_accu += _accu.detach().cpu().numpy()
        print("{} loss: {:.4f} and {} accuracy {:.2f}".format(str_use, val_loss / len(data_use)/len(_image),
                                                              str_use, val_accu / len(data_use) / len(_image)))
        return val_loss, val_accu / len(data_use) / len(_image)
                    
    def run(self):
        global_step = 0 
        for j in range(self.num_local_epochs):
            grad_group = []
            for i, (_im, _la) in enumerate(self.tr_data_loader):
                _grad_group = self._update_batch_tr(_im, _la, global_step)
                grad_group.append(_grad_group)
                global_step += 1 
                if global_step >= self.num_local_epochs * len(self.tr_data_loader):
                    _val_loss, _val_accu = self._eval(global_step, self.tt_data_loader, "test")
                    return self.model_use.state_dict()
                
            
def run_train(conf, tr_im, tr_la, local_id, exist_model, version=0):     
    print("===========================================================")
    print("                    Local ID %02d " % local_id)
    print("===========================================================")

    tt_im, tt_la = mnist_utils.load_tt_im()
    
    conf.batch_size = len(tr_im)
    print("The used batch size", conf.batch_size)
    tr_loader = mnist_utils.get_dataloader(tr_im, tr_la, True, conf.batch_size)
    tt_loader = mnist_utils.get_dataloader(tt_im, tt_la)

    train_obj = Train(conf, [tr_loader, tt_loader], conf.num_local_epochs, conf.sigma, exist_model, version)
    client_model = train_obj.run()
    print("Done Local ID %02d" % local_id )
    return client_model
    
    
def check_test_accuracy(model_checkpoints):
    tt_im, tt_la = mnist_utils.load_tt_im()
    tt_loader = mnist_utils.get_dataloader(tt_im, tt_la, shuffle=False, batch_size=100)
    model_use = mnist_utils.CLSModel(28*28, 10).to(device)        
    model_use.load_state_dict(model_checkpoints)
    loss, accu = 0.0, 0.0
    for i, (_im, _la) in enumerate(tt_loader):
        _im, _la = _im.to(device), _la.to(device)
        _pred = model_use(_im)
        _loss = nn.CrossEntropyLoss(reduction='sum')(_pred, _la) / len(_im) 
        _accu = (_pred.argmax(axis=-1) == _la).sum()
        loss += _loss.detach().cpu().numpy()
        accu += _accu.detach().cpu().numpy()
    loss = loss / len(tt_loader)
    accu = accu / len(tt_loader) / 100 
    print("Server model loss: %.4f and accuracy: %.4f" % (loss, accu)) 
    del model_use
    del _im 
    del _la 

   
def train_with_conf(conf):    
    stdoutOrigin = sys.stdout
    model_dir = "SOME NAME"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir += "version_%02d" % conf.version         
    conf.model_dir = model_dir 
    p_write = conf.model_dir + "SOME_NAME.txt"
    print("The directory", p_write)
    sys.stdout = open(p_write, 'w')

    tr_im, tr_la = mnist_utils.split_dataset_to_workers(conf.n_clients, conf.split)
    if conf.split != "iid" and conf.shuffle_percentage != 0:
        tr_im_sync, tr_la_sync, combine_im, combine_la = mnist_utils.shuffle_dataset(conf.n_clients, conf.shuffle_percentage, 
                                                                               )
    else:
        combine_im, combine_la = [tr_im["worker_%02d" % i] for i in range(10)], [tr_la["worker_%02d" % i] for i in range(10)]

    print("The updated number of local epochs", conf.num_local_epochs, conf.shuffle_percentage)
    print("The class frequency per client")
    for i in range(conf.n_clients):
        print("worker-%02d" % i, np.unique(combine_la[i], return_counts=True))
    init_time = time.time()
    print("Initial time: ", init_time)
    
    seed_use = np.random.randint(0,100000, 1)[0]
    print("The used learning rate", conf.lr)
    print("The seed", seed_use)
 
    for i in range(conf.num_rounds):
        if i == 0:            
            conf.random_state = np.random.RandomState(seed_use)
            mnist_utils.seed_everything(seed_use)
        conf.round = i
        if i == 0:
            exist_model = None 
        for j in range(conf.n_clients):
            c_tr_im, c_tr_la = combine_im[j], combine_la[j]
            _model = run_train(conf, c_tr_im, c_tr_la, j, exist_model, conf.version)
            if j == 0:
                model_group = {}
                for k in _model.keys():
                    model_group[k] = _model[k] * (1 / conf.n_clients)
            else:
                for k in _model.keys():
                    model_group[k] += _model[k] * (1 / conf.n_clients)
        exist_model = model_group 
        check_test_accuracy(exist_model)
            
    end_time = time.time()
    print("End time", end_time - init_time)
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    
    
if __name__ == "__main__":
    conf = give_args()
    train_with_conf(conf)

    

    
    



