import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np


class GREcA(nn.Module):
    def __init__(self, embedding_dim, group_user_matrix, user_item_matrix, group_item_bias, layers, drop_ratio):
        super(GREcA, self).__init__()
        self.embedding_dim = embedding_dim
        self.group_user_matrix = group_user_matrix
        self.user_item_matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]
        self.itemEmbedding = nn.Embedding(self.num_items, embedding_dim)
        self.group_item_bias = group_item_bias
        self.predictlayer = PredictLayer(2 * embedding_dim, layers, drop_ratio)


    def userEmbeds(self, item_embeds):
        assert(item_embeds.shape == (self.num_items, self.embedding_dim))

        mat1 = item_embeds.transpose(0, 1)
        mat2 = torch.FloatTensor(self.user_item_matrix.transpose().todense())

        user_embeds_intermediate = torch.mm(mat1, mat2).transpose(0, 1)
        sums = torch.gt(mat2, 0).float().sum(0)
        sums[sums < 1] = 1
        user_embeds = torch.div(user_embeds_intermediate.transpose(0, 1), sums)
        return user_embeds.transpose(0, 1)

    def groupEmbeds(self, user_embeds):
        mat1 = user_embeds.transpose(0, 1)
        mat2 = torch.FloatTensor(self.group_user_matrix.transpose().todense())

        group_embeds_intermediate = torch.mm(mat1, mat2).transpose(0, 1)
        sums = mat2.float().sum(0)
        group_embeds = torch.div(group_embeds_intermediate.transpose(0, 1), sums)
        return group_embeds.transpose(0, 1)

    def forward(self, group_inputs, item_inputs):
        assert(item_inputs.shape == group_inputs.shape)
        item_embeds = self.itemEmbedding(torch.LongTensor([i for i in range(self.num_items)]))
        user_embeds = self.userEmbeds(item_embeds)
        group_embeds = self.groupEmbeds(user_embeds)
        combined = torch.cat((group_embeds[group_inputs], item_embeds[item_inputs]), dim=1)
        y = torch.sigmoid(self.predictlayer(combined))
        return y


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, layers=[128, 8], drop_ratio=0):
        super(PredictLayer, self).__init__()

        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layers[i], layers[i+1]),
                    nn.ReLU(),
                    nn.Dropout(drop_ratio)
                )
            )
        self.layers.append(
                nn.Linear(layers[-1], 1)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
