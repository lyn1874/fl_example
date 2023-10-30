#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   get_subset_cifar10.py
@Time    :   2023/01/16 13:47:46
@Author  :   Bo 
'''
import torch 
import numpy as np 
import prepare_partition as pp 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch 
import os 


class Partition(object):
    def __init__(self, data, index):
        """Get a subset of data based on the index
        Args:
            data: object, full datset 
            index: index, full dataset
        """
        self.data = data 
        self.index = index 
        print(len(self.data), len(self.index))
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, sub_index):
        """Args:
        sub_index: the sub index for a particular partition of the dataset
        """
        data_idx = self.index[sub_index]
        return self.data[data_idx]      



class DataPartitioner(object):
    def __init__(self, data, partition_sizes, partition_type, 
                 partition_obj=True,
                 ):
        """Args:
        conf: the configuration arguments 
        data: Partition object or data array
        partition_sizes: number of data per device, [Number clients]
        partition_type: str
        consistent_indices: bool. If True, the indices are broadcast to all the devices        
        """
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        
        self.partitions = []
        
        if partition_obj == False: 
            self.data_size = len(data.targets)
            self.data = data 
            indices = np.array([x for x in range(self.data_size)])
        else:        
            self.data_size = len(data.index)
            self.data = data.data 
            indices = data.index 
        self.partition_indices(indices)
            
    def partition_indices(self, indices):
        indices = self._create_indices(indices)  # server create indices (I am not sure that I understand this part)
        from_index = 0 
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index         
    
    def _create_indices(self, indices):
        return indices

    def use(self, partition_id):
        return Partition(self.data, self.partitions[partition_id])


def define_train_dataset(client_id_index, num_clients, train_dataset, conf):
    
    partition_sizes = [
        1.0
    ]
    data_partitioner = DataPartitioner(
        train_dataset,
        partition_sizes,
        partition_type="original",
        partition_obj=False
    )
    tr_index = data_partitioner.partitions[0]
    tr_update_dataset = data_partitioner.use(0)
    print("get the data partitional")
    partition_size = [1.0 / num_clients for _ in range(num_clients)]
    assert client_id_index is not None
    data_partitioner = pp.DataPartitioner(conf, tr_update_dataset, 
                                          partition_sizes=partition_size,
                                          partition_type=conf.partition_type, 
                                          consistent_indices=False)
    print("prepared the data partitioner")
    data_to_load = data_partitioner.use(client_id_index)
    return data_to_load


def get_cifar(name, root="../image_dataset/", split="train", transform_apply=True):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    root: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    root = os.path.join(root, name)
    is_train = True if "train" in split else False

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )

    # decide data type.
    if is_train and transform_apply:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
    )
    
    

def get_cifar10_dataset(conf, transform_apply=True):
    train_dataset = get_cifar("cifar10", split="train", transform_apply=transform_apply)    
    print(conf.use_local_id)
    train_loader = define_train_dataset(conf.use_local_id, conf.n_clients, train_dataset, conf)
    shuffle=False if not transform_apply else True 
    data_loader = torch.utils.data.DataLoader(train_loader, 
                                        batch_size=conf.batch_size, 
                                        shuffle=shuffle, 
                                        num_workers=4, 
                                        pin_memory=True,
                                        drop_last=True)
    return data_loader 


def get_cifar10_test_dataset(batch_size):
    test_dataset = get_cifar("cifar10", split="test")
    data_loader = torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        num_workers=4, 
                                        pin_memory=True,
                                        drop_last=True)
    return data_loader
