import os
import pickle
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import pathlib
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader


def load_source_file(filename):
	'''
	load the file, return three array for constructing sparse matrix 
	col, row, data are the first, second, third columns
	'''
	df = pd.read_csv(filename)
	if len(df.columns) < 3:
		df['score'] = 1
	return df.values


def min_edge_filter(user_item, threshold=5):
	while True:
		user_count = {}
		for u, i, _ in user_item:
			if u not in user_count:
				user_count[u] = 0
			user_count[u] += 1
		item_count = {}
		for u, i, _ in user_item:
			if i not in item_count:
				item_count[i] = 0
			item_count[i] += 1
		user_item_ = [e for e in user_item if item_count[e[1]] >= threshold and user_count[e[0]] >= threshold]
		if len(user_item_) == len(user_item):
			return np.array(user_item)
		else:
			user_item = user_item_


def reindex(group_user, user_item, item_feature):
	#print(group_user)
	#print(user_item)
	groups = set(group_user[:, 0])
	users = set(group_user[:, 1]).union(user_item[:, 0])
	items = set(user_item[:, 1])
	group_hash = {g: i for i, g in enumerate(list(groups))}
	user_hash = {u: i for i, u in enumerate(list(users))}
	item_hash = {j: i for i, j in enumerate(list(items))}
	g_u = [[group_hash[e[0]], user_hash[e[1]], e[2]] for e in group_user]
	u_i = [[user_hash[e[0]], item_hash[e[1]], e[2]] for e in user_item]
	if item_feature is not None:
		i_f_data = []
		for e in item_feature:
			if e[0] in item_hash:
				i_f_data.append([item_hash[e[0]], e[1], e[2]])
		i_f = np.array(i_f_data)
		features = set(item_feature[:, 1])
	else:
		i_f = None
	# print(np.array(g_u).shape)
	# print(np.array(u_i).shape)
	# print(i_f.shape)
	return np.array(g_u), np.array(u_i), i_f, (len(groups), len(users), len(items), len(features))


def to_map(lst):
	m = {}
	for k, i, v in lst:
		if k not in m:
			m[k] = {}
		m[k][i] = v
	return m


def data_loader(X, Y, batch_size=256):
	data = TensorDataset(torch.FloatTensor(Y), torch.LongTensor(X))
	return DataLoader(data, batch_size=batch_size, shuffle=True)


def save(var, directory, name):
	path_to_file = os.path.join(directory, name)
	ext = name.split('.')[-1]
	if os.path.exists(path_to_file):
		return
	if ext == 'pkl':
		with open(path_to_file, 'wb') as f:
			pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
	elif ext == 'pth':
		torch.save(var, path_to_file)
	elif ext == 'npz':
		sp.save_npz(path_to_file, var)
	elif ext == 'csv':
		var.to_csv(path_to_file, index=False)
	else:
		assert(0)


def pivot_wide(lst, shape):
	r = [e[0] for e in lst]
	c = [e[1] for e in lst]
	v = [e[2] for e in lst]
	return sp.csr_matrix((v, (r, c)), shape=shape)


def mask(user_item, ratio=0.2):
	'''
	mask ratio of the masked user-item rating
	'''
	n_rows = user_item.shape[0]
	mask_indices = np.random.choice(n_rows, int(n_rows*ratio), replace=False)
	
	test_rows = user_item[mask_indices, :]
	train_rows = np.delete(user_item, mask_indices, axis=0)
	return np.array(test_rows), np.array(train_rows)


def calculate_prob_matrix(group_user_matrix, user_item_matrix):
	'''
	directly calculate the probability of g-i is metapath
	return the probability matrix
	'''
	group_item_prob = []
	for gid in range(group_user_matrix.shape[0]): # for each group
		users = list(group_user_matrix[gid, :].indices)
		group_rates = user_item_matrix[users, :].astype(float)
		for iid in range(group_rates.shape[1]): # for each item
			col = group_rates.getcol(iid)
			col_users = col.nonzero()[0]
			rates = col.data
			agree = 1 / (1 + np.var(rates)) if len(rates) != 0 else 0
			for user in col_users:
				group_rates[user, iid] = agree
		group_rates_normalized = normalize(group_rates, norm='l1', axis=1) / len(users)
		group_item_prob.append(group_rates_normalized.sum(axis=0).tolist()[0])
	return sp.csr_matrix(group_item_prob)


def calculate_guifi_prob(group_item_prob, item_feature_matrix):
	item_feature_prob = normalize(item_feature_matrix, norm='l1', axis=1)
	feature_item_prob = normalize(item_feature_matrix.transpose(), norm='l1', axis=1)
	print(group_item_prob.shape)
	print(item_feature_prob.shape)
	print(feature_item_prob.shape)

	group_feature = np.matmul(group_item_prob.toarray(), item_feature_prob.toarray())
	guifi = np.matmul(group_feature, feature_item_prob.toarray())
	return sp.csr_matrix(guifi)



def check_files(cacheDir, required_files):
	if not os.path.exists(cacheDir):
		return False
	for f in required_files:
		if not os.path.exists(os.path.join(cacheDir, f)):
			return False
	return True


def load(cacheDir, f):
	ext = f.split('.')[-1]
	path_to_file = os.path.join(cacheDir, f)
	if ext == 'pkl':
		with open(path_to_file, 'rb') as file:
			data = pickle.load(file)
		return data
	elif ext == 'pth':
		return torch.load(path_to_file)
	elif ext == 'npz':
		return sp.load_npz(path_to_file)
	elif ext == 'csv':
		return pd.read_csv(path_to_file)
	else:
		print(str(path_to_file), ' not found.')
		return 


def load_files(cacheDir, required_files):
	files = {}
	for f in required_files:
		files[f.split('.')[0]] = load(cacheDir, f)
	return files


def lr_decay(config, epoch_id):
	learning_rates = config.lr
	# learning rate decay
	lr = learning_rates[0]
	if epoch_id >= 15 and epoch_id < 25:
		lr = learning_rates[1]
	elif epoch_id >=20:
		lr = learning_rates[2]
	# lr decay
	if epoch_id % 5 == 0:
		lr /= 2
	return lr


def nDCG(rank, ratings):
    '''
    rank is a list of items in the order of group recommendation
    ratings is a dictionary with (key: item; value: rating) for a user
    '''

    def DCG(l):
        return l[0] + sum([l[i] / math.log2(i + 1) for i in range(1, len(l))])

    rank_score = [ratings[item] for item in rank if item in ratings]
    if len(rank_score) == 0:
        return 0
    dcg = DCG(rank_score)
    rank_score.sort(reverse=True)
    idcg = DCG(rank_score)
    if idcg == 0:
        return 0
    return dcg / idcg


def precision(rank, ratings):
    r = [ratings[i] for i in ratings]
    if len(r) < len(rank):
        true_positive = set([k for k in ratings])
    else:
        r.sort()
        threshold = r[-len(r)]
        true_positive = set([k for k in ratings if ratings[k] >= threshold])
    hit = set(rank).intersection(true_positive)
    return len(hit) / len(rank)


def recall(rank, ratings):
    true_positive = set([i for i in ratings])
    hit = set(rank).intersection(true_positive)
    return len(hit) / len(true_positive)


def setup(args):
	inputDir = args.input
	cacheDir = os.path.join(inputDir, 'cache_mpth' + str(args.mpth).split('.')[-1] + '/')
	outputDir = os.path.join(inputDir, args.output)
	model_name = args.model
	test_ep = set([int(e)-1 for e in args.test_ep])
	if args.logname == 'None':
		logname = model_name
	else:
		logname = args.logname
	pathlib.Path(outputDir).mkdir(parents=True, exist_ok=True)
	
	formal_out = Logger(os.path.join(outputDir, 'formal_' + logname))
	return inputDir, cacheDir, outputDir, model_name, test_ep, logname, formal_out


class Logger(object):
    """docstring for Loger"""
    def __init__(self, filename):
        super(Logger, self).__init__()
        self.logfile = filename
        with open(filename, 'w') as f:
            pass

    def log(self, s):
        with open(self.logfile, 'a') as f:
            f.write(s)

    def split(self):
        split = ''.join(['='] * 20) + '\n'
        self.log(split)

    def start_epoch(self, i):
        self.split()
        self.log('Start epoch ', i, '\n')

    def log_df(self, df):
        self.log(df.to_string() + '\n')
