import scipy.sparse as sp
import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm
import pickle

import random
import torch
import pathlib

import argparse
from util.utils import to_map, data_loader, load_source_file, \
						min_edge_filter, reindex, save, pivot_wide, \
						mask, calculate_prob_matrix, calculate_guifi_prob


class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, directory, min_deg=5, test_ratio=0.2, cache='cache/'):
		super(Dataset, self).__init__()
		# load the raw data
		groupUserFile = os.path.join(directory, 'groupUser.txt')
		userItemFile = os.path.join(directory, 'userItem.txt')
		itemFeatureFile = os.path.join(directory, 'itemFeature.txt')
		group_user = load_source_file(groupUserFile)
		user_item = load_source_file(userItemFile)
		if os.path.exists(itemFeatureFile):
			item_feature = load_source_file(itemFeatureFile)
		else:
			item_feature = None

		self.directory = cache if cache != 'cache/' else os.path.join(directory, cache)
		print('targetdir: ', self.directory)
		os.makedirs(self.directory, exist_ok=True)

		# reindex the data
		self.group_user, self.user_item, self.item_feature, self.shapes = reindex(group_user, min_edge_filter(user_item, threshold=min_deg), item_feature)

		# mask some user-item ratings
		self.test_user_item, self.train_user_item = mask(self.user_item, ratio=test_ratio)


	def save_mpe_files(self, threshold=0.5, relative=True, w=1):
		'''
		1. group user matrix 
		2. train user item matrix 
		3. train data loader using agreement biased meta path
		4. Is the matrix necessary? change to map?
		'''
		# check existance
		files = [os.path.join(self.directory, f) for f in ['mpe_train_group_user.npz', \
															'mpe_train_user_item.npz', \
															'mpe_train_group_item.pth', \
															'metapath.csv', \
															'group_agreed_item,npz']]
		if [os.path.exists(p) for p in files] == [True, True, True]:
			print('already have mpe files, use backup')
			return

		# save the sparse matrices
		mpe_group_user_matrix = pivot_wide(self.group_user, (self.shapes[0], self.shapes[1]))
		mpe_user_item_matrix = pivot_wide(self.train_user_item, (self.shapes[1], self.shapes[2]))
		if self.item_feature is not None:
			mpe_item_feature_matrx = pivot_wide(self.item_feature, (self.shapes[2], self.shapes[3]))
		else:
			mpe_item_feature_matrx = None
		save(mpe_group_user_matrix, self.directory, 'mpe_train_group_user.npz')
		save(mpe_user_item_matrix, self.directory, 'mpe_train_user_item.npz')	

		# construct the train data loader
		if os.path.exists(os.path.join(self.directory, 'group_item_prob.npz')):
			group_item_prob = sp.load_npz(os.path.join(self.directory, 'group_item_prob.npz'))
		else:
			group_item_prob = calculate_prob_matrix(mpe_group_user_matrix, mpe_user_item_matrix)
			save(group_item_prob, self.directory, 'group_item_prob.npz')
		if os.path.exists(os.path.join(self.directory, 'group_item_prob_long.npz')):
			guifi_prob = sp.load_npz(os.path.join(self.directory, 'group_item_prob_long.npz'))
		else:
			guifi_prob = calculate_guifi_prob(group_item_prob, mpe_item_feature_matrx)
			save(guifi_prob, self.directory, 'group_item_prob_long.npz')
		

		def get_mp(mat):
			metapath = []
			for gid in range(mat.shape[0]): # for each group
				items = list(mat[gid, :].indices)
				probs = list(mat[gid, :].data)
				if len(probs) == 0:
					continue
				bar = threshold * max(probs) if relative else threshold
				for iid, prob in zip(items, probs):
					metapath.append([gid, iid, int(prob > bar), prob])
			return metapath

		# metapath = get_mp(group_item_prob)
		# if guifi_prob is not None:
		# 	metapath += get_mp(guifi_prob)

		metapath = get_mp(group_item_prob * w + guifi_prob * (1-w))
		print('weight for short: ', w)

		df = pd.DataFrame(metapath, columns=['group', 'item', 'label', 'prob'])
		save(df, self.directory, 'metapath.csv')

		group_agreed_item = df[df['label'] == 1].values[:, :3].astype(int)
		a = pivot_wide(group_agreed_item, (self.shapes[0], self.shapes[2]))
		save(a, self.directory, 'group_agreed_item.npz')
		
		positive = [mp for mp in metapath if mp[2] == 1]
		negative = [mp for mp in metapath if mp[2] == 0]
		num_neg = min(len(positive), len(negative))
		
		selected = random.sample(negative, k=num_neg)
		mpe_train_group_item = np.array(positive+selected)
		print('num_neg: ', num_neg, '; ', mpe_train_group_item.shape)
		np.random.shuffle(mpe_train_group_item)
		save(data_loader(mpe_train_group_item[:, :2], mpe_train_group_item[:, 2]), self.directory, 'mpe_train_group_item.pth')


	def save_test_files(self):
		'''
		1. test loaders
		2. origin_user_rating for verification usage
		'''
		# check existance
		files = [os.path.join(self.directory, f) for f in ['test.pth', \
															'original_user_item_map.pkl', \
															'original_group_user_map.pkl']]
		if [os.path.exists(p) for p in files] == [True, True, True]:
			print('already have test files, use backup')
			return

		# construct and save the test loader
		# g, i in a test loader only if the item is no seen for all user in the group 
		test = []
		test_user_item_map = to_map(self.test_user_item)
		train_user_item_map = to_map(self.train_user_item)
		for g, users in to_map(self.group_user).items():
			test_items = set()
			train_items = set()
			for u in users:
				if u in test_user_item_map:
					test_items = test_items.union(test_user_item_map[u].keys())
				if u in train_user_item_map:
					train_items = train_items.union(train_user_item_map[u].keys())

			pos_test_items = test_items - train_items
			all_neg = set([i for i in range(self.shapes[2])]) - train_items - test_items
			if len(pos_test_items) > 0:
				neg_test_items = random.sample(all_neg, k=min(100, len(all_neg)))
				test += [[g, i, 1] for i in set(pos_test_items)]
				test += [[g, i, 0] for i in neg_test_items]
		test = np.array(test)
		np.random.shuffle(test)
		save(data_loader(test[:, :2], test[:, 2]), self.directory, 'test.pth')
		
		# original user rating for verification purpose
		save(to_map(self.user_item), self.directory, 'original_user_item_map.pkl')

		# save the group user map
		print(self.group_user)
		save(to_map(self.group_user), self.directory, 'original_group_user_map.pkl')

		# construct the true values for test file
		majority_vote = test_majority_vote(to_map(self.group_user), test_user_item_map)
		save(majority_vote, self.directory, 'majority_vote_result.pkl')


def preprocess(inputDir, min_deg=5, test_ratio=0.2, mp_threshold=0.5, cache='cache/', w=1):
	dataset = Dataset(inputDir, min_deg=min_deg, test_ratio=test_ratio, cache=cache)
	dataset.save_mpe_files(threshold=mp_threshold, relative=True, w=w)
	dataset.save_test_files()



