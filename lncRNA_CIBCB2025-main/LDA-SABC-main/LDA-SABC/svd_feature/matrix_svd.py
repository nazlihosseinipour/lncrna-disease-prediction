# encoding=utf-8

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy import sparse as sp

pca_dim = 5
pca_maxiter = 200


def vector_to_diagonal(vector):
    """
    将向量放在对角矩阵的对角线上
    :param vector:
    :return:
    """
    if (isinstance(vector, np.ndarray) and vector.ndim == 1) or \
            isinstance(vector, list):
        length = len(vector)
        diag_matrix = np.zeros((length, length))
        np.fill_diagonal(diag_matrix, vector)
        return diag_matrix
    return None


interMatrix = pd.read_excel('../data2/MNDR-lncRNA-disease associations matrix.xls',header=0,index_col=0).values

interMatrix = interMatrix.astype('float')
U, S, VT = svds(sp.csr_matrix(interMatrix), k=pca_dim, maxiter=pca_maxiter)
S = vector_to_diagonal(S)

print('RNA vector representation shape:')
print(U.shape)
print('Singular value matrix：')
print(np.sum(S, axis=0))
print('disease vector representation shape:')
print(VT.T.shape)

np.save('../data2/r5_feature.npy', U)
np.save('../data2/d5_feature.npy', VT.T)

