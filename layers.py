import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
from keras.src.layers import LayerNormalization
from keras import layers

from utils import *


#GAT代码
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903. This part of code refers to the implementation of https://github.com/Diego999/pyGAT.git

    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # （N，N）

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return F.relu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#自注意力机制
class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output

#GCN代码
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        output = torch.matmul(adj, Wh)
        return F.relu(output)

#GAC代码
class GraphAttentionLayer_GAC(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer_GAC, self).__init__()
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAC, self).__init__()
        self.gc1 = GraphAttentionLayer_GAC(input_dim, hidden_dim)
        self.gc2 = GraphAttentionLayer_GAC(hidden_dim, output_dim)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        x = self.gc2(x, adj)
        return x


class GraphAttentionBiLSTMConvolution(nn.Module):

    #初始化构造函数，接收输入维度、输出维度、名称、dropout 比率、拼接选项和激活函数。
    def __init__(self, in_features, out_features):
        #设置类属性，初始化变量字典和稀疏标志。
        self.input_dim=in_features
        self.output_dim=out_features
        self.vars = {}
        self.issparse = False
        #使用 Glorot 初始化方法初始化权重。
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                in_features, out_features, name='weights')

        #设置 alpha 值、拼接选项，并定义双向 LSTM 层。
        self.alpha = 0.2
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(units=int(self.output_dim/2), input_shape=(10, self.input_dim))
        )
        #初始化权重矩阵和注意力系数矩阵。
        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(in_features, out_features)))
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * out_features, 1)))
        #定义激活函数和层归一化。
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.layer = LayerNormalization(axis=1)

    #定义前向传播方法，接受输入 h 和邻接矩阵 adj。
    def __call__(self, h, adj, training=True):
        # 扩展输入维度并通过双向 LSTM 层处理。
        with tf.compat.v1.name_scope(self.name):
            h = tf.expand_dims(h, axis=-1)
            h = self.bi_lstm(h)
            # 计算加权输入和准备注意力机制的输入。
            Wh = tf.matmul(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            #处理邻接矩阵，确保其为密集形式并添加单位矩阵。
            zero_vec = -9e15 * tf.ones_like(e)
            adj=tf.sparse.to_dense(tf.sparse.reorder(adj))
            adj = adj + tf.eye(tf.shape(adj)[0])
            #计算注意力权重并应用 dropout。
            attention = tf.where(adj > 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            #根据注意力权重计算输出并应用激活函数。
            h_prime = tf.matmul(attention, Wh)
        return self.leakyrelu(h_prime)

    #准备注意力机制的输入，计算注意力分数。