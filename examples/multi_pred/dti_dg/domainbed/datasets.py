# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os
import torch
import numpy as np
import pandas as pd
import pickle
import json

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 0            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


# DeepDTA里amino_char和smiles_char都没有问号，分别是25和62个
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

from sklearn.preprocessing import OneHotEncoder
# 独热编码
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

# 这里把smiles串转成了长为100的字符串，不在smiles_char list里的用？表示
def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

from torch.utils import data

class dti_tensor_dataset(data.Dataset):

    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        d = self.df.iloc[index].Drug_Enc
        t = self.df.iloc[index].Target_Enc
        
        d = drug_2_embed(d)
        t = protein_2_embed(t)
        
        y = self.df.iloc[index].Y
        return d, t, y

class TdcDtiDg(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        
        ENVIRONMENTS = [str(i) for i in list(range(2013, 2022))]
        
        # 查看了数据集，实际上train只有2013-2018，test只有2019-2021，所以这里如此设定
        TRAIN_ENV = [str(i) for i in list(range(2013, 2019))]
        TEST_ENV = ['2019', '2020', '2021']
        
        #TRAIN_ENV = ['2019', '2020']
        #TEST_ENV = ['2021']
        
        # TODO: 这里的shape代表什么呢，100是drug的长度，1000是氨基酸链的长度，26和63是什么呢？
        # 懂了，26是amino_char的长度，63是smiles_char的长度,那为什么是(26, 100), (63, 1000)呢？
        self.input_shape = [(26, 100), (63, 1000)]
        self.num_classes = 1
        
        if root is None:
            raise ValueError('Data directory not specified!')
       
        ## create a datasets object
        self.datasets = []
        from tdc import BenchmarkGroup
        self.group = BenchmarkGroup(name = 'DTI_DG_Group', path = root)
        benchmark = self.group.get('BindingDB_Patent') 
        # 这里把所有的数据拿到
        train_val, test, name = benchmark['train_val'], benchmark['test'], benchmark['name']
        
        """
        Pandas模块的数据结构主要有两种:
        1.Series 2.DataFrame
        Series 是一维数组, 基于Numpy的ndarray 结构
        """
        unique_drug = pd.Series(train_val['Drug'].unique()).apply(trans_drug)
        # unique_dict_drug里，smiles字符串是key
        unique_dict_drug = dict(zip(train_val['Drug'].unique(), unique_drug))
        keys = unique_dict_drug.keys()
        f = open('train_smiles.txt','a')
        for i in keys:    
            f.write(i+"\n")
        f.close()
        train_val['Drug_Enc'] = [unique_dict_drug[i] for i in train_val['Drug']]

        unique_target = pd.Series(train_val['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(train_val['Target'].unique(), unique_target))
        train_val['Target_Enc'] = [unique_dict_target[i] for i in train_val['Target']]

        for i in TRAIN_ENV:
            df_data = train_val[train_val.Year == int(i)]
            self.datasets.append(dti_tensor_dataset(df_data))
            print('Year ' + i + ' loaded...')

        unique_drug = pd.Series(test['Drug'].unique()).apply(trans_drug)
        unique_dict_drug = dict(zip(test['Drug'].unique(), unique_drug))
        keys = unique_dict_drug.keys()
        f = open('test_smiles.txt','a')
        for i in keys:    
            f.write(i+"\n")
        f.close()
        test['Drug_Enc'] = [unique_dict_drug[i] for i in test['Drug']]

        unique_target = pd.Series(test['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(test['Target'].unique(), unique_target))
        test['Target_Enc'] = [unique_dict_target[i] for i in test['Target']]
        
        for i in TEST_ENV:
            df_data = test[test.Year == int(i)]
            self.datasets.append(dti_tensor_dataset(df_data))
            print('Year ' + i + ' loaded...')
