import sys
import pickle

import argparse
import scipy.sparse as sp
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from statistics import mean
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


from config import Config
from preprocess import preprocess
from models.GREcA import GREcA
from util.utils import check_files, load_files, lr_decay, Logger, setup
from util.tester import Tester


required_files = [
    'mpe_train_user_item.npz',
    'original_group_user_map.pkl',
    'group_item_prob.npz',
    'original_user_item_map.pkl',
    'mpe_train_group_item.pth',
    'test.pth',
    'mpe_train_group_user.npz',
    'metapath.csv',
    'group_agreed_item.npz'
]


def ini_model(model_name, data, config):
    num_group, num_user = data['mpe_train_group_user'].get_shape()
    num_user_, num_item = data['mpe_train_user_item'].get_shape()
    return GREcA(config.embedding_size, data['mpe_train_group_user'], data['mpe_train_user_item'], data['group_agreed_item'], config.layers, config.drop_ratio)


def train_mpe(model, dataloader, lr):
    optimizer = optim.Adam(model.parameters(), lr)
    losses = []
    for _, (y, X) in tqdm(enumerate(dataloader)):
        group_input = torch.LongTensor(X[:, 0])
        item_input = torch.LongTensor(X[:, 1])
        # Forward
        prediction = model(group_input, item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((prediction.flatten() - y) **2)
        # record loss history
        losses.append(loss)
        # Backward
        loss.backward()
        optimizer.step()
    return torch.mean(torch.stack(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input directory")
    parser.add_argument("--output", type=str, default='out/', help="output directory")
    parser.add_argument("--model", type=str, default='mpe')
    parser.add_argument("--mpth", type=float, default=0.5)
    parser.add_argument("--logname", type=str, default='None')
    parser.add_argument("--w", type=float, default=1)
    parser.add_argument('-t', "--test_ep", action='append', default=[50])


    inputDir, cacheDir, outputDir, model_name, test_ep, logname, formal_out = setup(parser.parse_args())
    config = Config()

    # check files
    if not check_files(cacheDir, required_files):
        preprocess(inputDir, min_deg=5, test_ratio=0.2, mp_threshold=parser.parse_args().mpth, cache=cacheDir, w=parser.parse_args().w)
    data = load_files(cacheDir, required_files)

    
    testers = [Tester(data['test'], data['original_user_item_map'], \
            data['original_group_user_map'], outputDir, k=k) for k in config.ks]

    model = ini_model(model_name, data, config)

    # config information
    print(model_name, " at embedding size %d, run Iteration:%d" %(config.embedding_size, config.epoch))
    # train the model
    for epoch in range(config.epoch):
        model.train()
        print('epoch: ', epoch)

        lr = lr_decay(config, epoch)
        loss = [train_mpe(model, data['mpe_train_group_item'], lr)]
        if epoch in test_ep:
            formal_out.log('Epoch ' + str(epoch) + ':\n')
            [tester.test_all(model, model_name, formal_out) for tester in testers]

    [tester.test_all(model, model_name, formal_out) for tester in testers]
    print("Done!")