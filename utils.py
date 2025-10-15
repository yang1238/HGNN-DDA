import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import tensorflow as tf

def multiomics_data():

    #protein_fea
    PHIN = np.genfromtxt("./2feature/disease_hin2vec.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    ESM = np.genfromtxt("./2feature/disease_Word2Vec.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    PHIN = np.array(PHIN)
    PHIN = scale(np.array(PHIN[:, 1:], dtype=float))
    ESM = np.array(ESM)
    ESM = scale(np.array(ESM[:, 1:], dtype=float))
    ppi_adj = get_adj_array("./0data/diseasedisease.txt")


    DHIN = np.genfromtxt("./2feature/drug_features_hin2vec.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    PC = np.genfromtxt("./2feature/drug_Physical_chemistry.csv", delimiter=',',skip_header=1)
    DHIN = np.array(DHIN)
    DHIN = scale(np.array(DHIN[:, 1:], dtype=float))
    PC = np.array(PC)
    PC = np.nan_to_num(PC,nan=0.0)
    PC = scale(np.array(PC[:, 1:], dtype=float))

    ddi_adj = get_adj_array("./0data/drugdrug.txt")

    # 加载label
    labellist = []
    with open('./0data/label.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # 移除行首和行尾的空白字符
        elements = line.split(" ")  # 使用空格分隔元素
        processed_elements = [int(elements[1]), int(elements[0]), int(elements[2])]  # 互换位置
        labellist.append(processed_elements)
    labellist = torch.Tensor(labellist)
    print("drug protein lable:", labellist.shape)


    protein_HIN, protein_ESM, protein_adj = torch.FloatTensor(PHIN), torch.FloatTensor(ESM), torch.FloatTensor(ppi_adj)
    drug_HIN, drug_PC, drug_adj = torch.FloatTensor(DHIN), torch.FloatTensor(PC), torch.FloatTensor(ddi_adj)
    return protein_HIN, protein_ESM, protein_adj, drug_HIN, drug_PC, drug_adj, labellist


def get_adj_array(file_path):
    # 读取txt文件
    file_path = file_path  # 替换为你的文件路径
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取邻接矩阵信息
    adj_matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]  # 假设邻接矩阵中的元素以空格分隔
        adj_matrix.append(row)

    # 将邻接矩阵转换为NumPy数组
    adj_array = np.array(adj_matrix)
    return adj_array


def weight_variable_glorot(input_dim, output_dim, name=""):
    #计算初始化范围 init_range，以保证权重在合理区间内。
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    #使用均匀分布初始化权重矩阵，范围在 [-init_range, init_range] 之间。
    initial = tf.compat.v1.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.compat.v1.float32
    )
    #返回一个 TensorFlow 变量，表示初始化后的权重矩阵。
    return tf.compat.v1.Variable(initial, name=name)
