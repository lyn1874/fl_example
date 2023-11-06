#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/03/05 15:26:47
@Author  :   Bo 
'''
import numpy as np 
from scipy.special import softmax 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import random 
import os 
import configs 
import get_subset_cifar10 as gsc 


mnist_path = "../image_dataset/"

def get_mnist_transform():
    normalize = (
        transforms.Normalize((0.1307,), (0.3081,))
    )
    transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    return transform


def get_one_hot(ylabel, n_class):
    y_label_one_hot = np.zeros([len(ylabel), n_class])
    for i, s_la in enumerate(ylabel):
        y_label_one_hot[i, int(s_la)] = 1
    return y_label_one_hot


def get_cross_entropy(prediction, ylabel, epsilon=1e-9):
    """Args:
    prediction: [num_samples, num_classes]
    ylabel: [num_samples, 1]    
    Ops:
        1. apply softmax on the prediction 
        2. one hot encoding ylabel 
        3. calculate the cross entropy loss 
    """
    soft_pred = softmax(prediction, axis=-1)
    predictions = np.clip(soft_pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(ylabel*np.log(predictions+1e-9))/N
    return ce

    
def verify_implementation():
    ylabel = np.random.randint(0, 10, [20])
    pred = np.random.random([20, 10])
    ce_numpy = get_cross_entropy(pred, get_one_hot(ylabel, 10))
    
    ce_torch = nn.CrossEntropyLoss(reduction='sum')(torch.from_numpy(pred).to(torch.float32), 
                                                    torch.from_numpy(ylabel).to(torch.int64))/len(pred)
    return ce_numpy, ce_torch 


def get_subset_images(train_images, train_label, num_select, save=False, tds_dir="../../image_dataset/"):
    """Prepare the subset of mnist
    train_dataset = prepare_mnist.get_dataset(conf, "mnist", "../image_dataset/", split="train")
    test_dataset = prepare_mnist.get_dataset(conf, "mnist", "../image_dataset/", split="test")
    """
    train_images = train_images.detach().numpy()
    train_label = train_label.detach().numpy()
    # np.random.seed(102)
    index_group = []
    for i in np.unique(train_label):
        _index = np.random.choice(np.where(train_label == i)[0], num_select, replace=False)
        index_group.append(_index)
    train_im_subset = train_images[np.reshape(index_group, [-1])]
    train_im_subset = np.reshape(train_im_subset, [len(train_im_subset), -1]).astype(np.float32) / 255.0
    train_la_subset = train_label[np.reshape(index_group, [-1])]
    print("The class distribution in the subset", np.unique(train_la_subset, return_counts=True))
    if save:
        np.savez(tds_dir + "/mnist_subset_%d.npz" % num_select, train_im_subset, train_la_subset)


def split_dataset_to_workers(num_workers, split):
    tr_dataset = np.load(mnist_path + "/mnist_subset_1024.npz")
    tr_im = tr_dataset["arr_0"]
    tr_la = tr_dataset["arr_1"]
    tr_im_clients = {}
    tr_la_clients = {}

    if split == "by_class":
        unique_cls = np.unique(tr_la)
        for i, s_cls in enumerate(unique_cls):
            _index = np.where(tr_la == s_cls)[0]
            tr_im_clients["worker_%02d" % i] = np.reshape(tr_im[_index], [len(_index), 28, 28, 1])
            tr_la_clients["worker_%02d" % i] = tr_la[_index]
    elif split == "iid":
        shuffle_index = np.random.choice(np.arange(len(tr_la)), len(tr_la), replace=False)
        tr_la_shuffle_index = np.split(shuffle_index, num_workers)
        for i in range(num_workers):
            tr_im_clients["worker_%02d" % i] = np.reshape(tr_im[tr_la_shuffle_index[i]], [len(tr_la_shuffle_index[i]), 28, 28, 1])
            tr_la_clients["worker_%02d" % i] = tr_la[tr_la_shuffle_index[i]]
    return tr_im_clients, tr_la_clients 


def shuffle_dataset(num_workers, p):
    if p == 0:
        return [[] for _ in range(num_workers)], [[] for _ in range(num_workers)]
    tr_dataset = np.load(mnist_path + "/mnist_subset_1024.npz")
    
    tr_im = tr_dataset["arr_0"]
    tr_la = tr_dataset["arr_1"]
    num_shuffle = int(p * 1024)
    # num_shuffle = int(np.ceil(p * 1024 / (1-p)))
    p_im, p_la = [], []
    r_im, r_la = [], []
    np.random.seed(1024)
    for i in np.unique(tr_la):
        index = np.where(tr_la == i)[0]
        select = np.random.choice(np.arange(len(index)), num_shuffle, replace=False)
        real_index = np.delete(np.arange(len(index)), select)
        p_im.append(tr_im[index[select]])
        p_la.append(tr_la[index[select]])
        r_im.append(tr_im[index[real_index]])
        r_la.append(tr_la[index[real_index]])

    p_im = np.concatenate(p_im, axis=0)
    p_la = np.concatenate(p_la, axis=0)
    shuffle_index = np.random.choice(np.arange(len(p_im)), len(p_im), replace=False)
    p_im = p_im[shuffle_index]
    p_la = p_la[shuffle_index]
    split_index = np.split(np.arange(len(p_im)), num_workers)
    
    p_im_client_base = [np.reshape(p_im[v], [len(v), 28, 28, 1]) for v in split_index]
    p_la_client_base = [p_la[v] for v in split_index]
    
    combine_im, combine_la = [], []
    for i in np.unique(tr_la):
        combine_im.append(np.concatenate([np.reshape(r_im[i], [len(r_im[i]), 28, 28, 1]), 
                                          p_im_client_base[i]], axis=0))
        combine_la.append(np.concatenate([r_la[i], p_la_client_base[i]], axis=0))
    
    return p_im_client_base, p_la_client_base, combine_im, combine_la 


def combine_real_shuffle(r_im, r_la, s_im, s_la, p):
    if p == 0:
        return r_im, r_la 
    else:
        c_im = np.concatenate([r_im, s_im], axis=0)
        c_la = np.concatenate([r_la, s_la], axis=0)
        print("shuffle percentage", len(s_im) / (len(s_im) + len(r_im)))
        return c_im, c_la


def load_tt_im():
    tt_dataset = np.load(mnist_path + "/mnist_test.npz")
    tt_im = np.reshape(tt_dataset["arr_0"], [-1, 28, 28, 1])
    tt_la = tt_dataset["arr_1"]
    return tt_im, tt_la 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

class GetData(object):
    def __init__(self, data, targets):
        """Get a subset of data based on the index
        Args:
            data: object, full datset 
            index: index, full dataset
        """
        self.transform = get_mnist_transform()
        self.data = data 
        self.targets = targets 
        print("The shape of the dataset", np.shape(self.data), np.shape(self.targets))
        self.index = np.arange(len(self.data))
        print("The length of the dataset", len(self.index))
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, sub_index):
        """Args:
        sub_index: the sub index for a particular partition of the dataset
        """
        data_idx = self.index[sub_index]
        _data = self.data[data_idx]
        if self.transform:
            _data = self.transform(_data)
        return _data, self.targets[self.index[sub_index]]    


def get_dataloader(im, label, shuffle=True, batch_size=None):
    data_to_load = GetData(im, label)
    batch_size = len(im) if not batch_size else batch_size
    data_loader = torch.utils.data.DataLoader(data_to_load, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=4, 
                                            pin_memory=True,
                                            drop_last=False)
    return data_loader
    
    
def create_model(conf):
    num_input = 28 * 28 if conf.dataset == "mnist" else 3 * 32 * 32 
    num_class = 10 
    num_channel = 1 if conf.dataset == "mnist" else 3 
    if conf.model_type == "s_mlp":
        model_use = CLSModel(num_input, num_class)
    elif conf.model_type == "m_mlp":
        model_use = CLSMultiLayerModel(num_input)
    elif conf.model_type == "m_cnn":
        model_use = CNNModel(num_channel)
    return model_use     
    

class CLSModel(nn.Module):
    def __init__(self, num_input, num_class):
        super(CLSModel, self).__init__()
        self.fc_layer = nn.Linear(num_input, num_class)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)
    
    
class CLSMultiLayerModel(nn.Module):
    def __init__(self, num_input):
        super(CLSMultiLayerModel, self).__init__()
        self.num_input = num_input
        self.fc_layer = nn.Sequential(nn.Linear(num_input, 200), 
                                      nn.ReLU(True), 
                                      nn.Linear(200, 200), 
                                      nn.ReLU(True), 
                                      nn.Linear(200, 10))
    def forward(self, x):
        x = x.view(x.size(0), self.num_input)
        return self.fc_layer(x)
    
    
class CNNModel(nn.Module):
    def __init__(self, num_channel):
        super(CNNModel, self).__init__()
        num_feat = 64 * 5 * 5 if num_channel == 3 else 1024 
        self.num_feat = num_feat 
        self.layer = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True))
        self.cls_layer = nn.Sequential(
            nn.Linear(num_feat, 512),
            nn.ReLU(True),
            nn.Linear(512, 10))
    def forward(self, x):
        feat = self.layer(x)
        out = self.cls_layer(feat.view(len(x), self.num_feat))
        return out 
        
    
    
def initial_model(conf):
    model = create_model(conf).to(torch.device("cuda"))
    model_param = {}
    for name, p in model.named_parameters():
        model_param[name] = p 
    return model_param


def define_optimizer(model, lr=None):
    # define the param to optimize.
    lr_group = {}
    momentum_group = {}
    weight_decay_group = {}
    for key, value in model.named_parameters():
        lr_group[key] = lr 
        momentum_group[key] = 0
        weight_decay_group[key] = 0    
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": weight_decay_group[key],
            "param_size": value.size(),
            "nelement": value.nelement(),
            "lr": lr_group[key],
            "momentum": momentum_group[key],
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    optimizer = torch.optim.SGD(
        params,
        nesterov=False,
    )
    return optimizer


    

def create_dir(conf):
    model_mom = "SOME NAME"  
    model_dir = model_mom + "SOME NAME" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
        

    
def create_cifar10_dir(conf):
    model_mom = "../exp_data/"
    model_dir = model_mom + "/cifar10/version_0/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    conf.folder_name = "cifar10" 
    conf.dir_name = "version_0"
    return conf      

    
if __name__ == "__main__":
    conf = configs.give_args()
    if conf.dataset == "mnist":
        create_dir(conf)
    elif conf.dataset == "cifar10":
        conf = create_cifar10_dir(conf)
        seed_use = np.random.randint(0,100000, 1)[0]
        conf.random_state = np.random.RandomState(seed_use)
        tr_loader = gsc.get_cifar10_dataset(conf, transform_apply=True)

    
        
        






        
    

