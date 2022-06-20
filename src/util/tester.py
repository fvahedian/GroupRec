import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from statistics import mean
from util.utils import nDCG, precision, recall, Logger
import json
import math
import os


class Tester(object):
    def __init__(self, test_loader, user_ratings, group_users, folder, k=10):
        super(Tester, self).__init__()
        self.test_loader = test_loader
        self.user_ratings = user_ratings
        self.group_users = group_users
        self.result_map = {}
        self.k = k
        self.log = folder + 'logv'
        self.cache_str = ''


    def test_results(self, model, model_name):
        '''
        return n_test * 3 np array
        each row is group id, item id, prediction
        '''
        results = []
        for batch_id, (y, X) in tqdm(enumerate(self.test_loader)):
            group_input = torch.LongTensor(X[:, 0])
            item_input = torch.LongTensor(X[:, 1])
            prediction = model(group_input, item_input).detach().numpy()
            results += np.hstack((
                np.reshape(group_input.numpy(), (-1, 1)),
                np.reshape(item_input.numpy(), (-1, 1)),
                prediction.reshape((-1, 1))
                )).tolist()
        self.result_map = {g: {'entire': []} for g in set([e[0] for e in results])}
        for result in results:
            self.result_map[result[0]]['entire'].append([result[1], result[2]])

        for g, result in self.result_map.items():
            np_pred = np.array(result['entire'])
            np_pred = np_pred[(-1 * np_pred[:, 1]).argsort()]
            self.result_map[g]['selected'] = np_pred[:self.k, :].tolist()

        return results

    def log_user_results(self, rank_item, u, res):
        res_info = ['ndcg: ', str(res[0]), '\t\trecall: ', str(res[1]), '\t\tprecision: ', str(res[2]), '\n']
        self.cache_str += '\tuser ' + str(u) + ' results: ' + ''.join(res_info)
        user_ratings = self.user_ratings[u]
        target = set(rank_item)
        for i in user_ratings:
            if i in target:
                self.cache_str += '\t\titem ' + str(i) + ': ' + str(user_ratings[i]) + '\n'
        self.cache_str += '\n'


    def avg_metric(self, metric):
        avg_results = []
        for g, result in self.result_map.items():
            rank_item = [p[0] for p in result['selected']]
            users = self.group_users[g]
            for u in users:
                if u not in self.user_ratings:
                    print(g, u)
            user_results_for_g = {u: [
                nDCG(rank_item, self.user_ratings[u]), 
                recall(rank_item, self.user_ratings[u]), 
                precision(rank_item, self.user_ratings[u])] for u in users}
            if self.v:
                self.cache_str += 'Recommended items for group ' + str(g) + ': ' + str(rank_item) + '\n'
                [self.log_user_results(rank_item, u, user_results_for_g[u]) for u in users]
            mean_ndcg = mean([user_results_for_g[u][0] for u in user_results_for_g])
            mean_recall = mean([user_results_for_g[u][1] for u in user_results_for_g])
            mean_precision = mean([user_results_for_g[u][2] for u in user_results_for_g])
            avg_results.append([g, mean_ndcg, mean_recall, mean_precision])
        return avg_results


    def test_all(self, model, model_name, formal):
        results = self.test_results(model, model_name)
        all_results = pd.DataFrame(self.avg_metric('_'), columns=['group', 'ndcg', 'recall', 'precision'])
        avg_ndcg = mean(all_results['ndcg'])
        avg_recall = mean(all_results['recall'])
        avg_precision = mean(all_results['precision'])
        formal.log('\n ' + model_name + '\n\
            Test Results top ' + str(self.k) + ': \n\
            \tndcg: ' + str(avg_ndcg) + '\n\
            \trecall: ' + str(avg_recall) + '\n\
            \tprecision: ' + str(avg_precision) + '\n')
        self.cache_str = ''