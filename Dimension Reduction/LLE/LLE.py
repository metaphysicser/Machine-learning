#-*-coding:UTF-8-*-
"""
@date:2020 8.6 21:07
@author:张平路
@function:realize LLE Algorithm
"""
#仍然不太明白，回头再看
import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
"""
@function:calculate the distance between two points
@parameter:data
@return:dist
"""
def cal_pairwise_dist(data):
    expand_ = data[:, np.newaxis, :]
    repeat1 = np.repeat(expand_, data.shape[0], axis=1)
    repeat2 = np.swapaxes(repeat1, 0, 1)
    D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
    return D

def get_n_neighbors(data, n_neighbors=10):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    N = np.zeros((n, n_neighbors))
    for i in range(n):
        # np.argsort 列表从小到大的索引
        index_ = np.argsort(dist[i])[1:n_neighbors +1]
        N[i] = N[i] + index_
    return N.astype(np.int32)

def lle(data, n_dims=2, n_neighbors=10):
	N = get_n_neighbors(data, n_neighbors)            # k近邻索引
	n, D = data.shape                                 # n_samples, n_features
	# prevent Si to small
	if n_neighbors > D:
		tol = 1e-3
	else:
		tol = 0
	# calculate W
	W = np.zeros((n_neighbors, n))
	I = np.ones((n_neighbors, 1))
	for i in range(n):                                # data[i] => [1, n_features]
		Xi = np.tile(data[i], (n_neighbors, 1)).T     # [n_features, n_neighbors]
		                                              # N[i] => [1, n_neighbors]
		Ni = data[N[i]].T                             # [n_features, n_neighbors]
		Si = np.dot((Xi-Ni).T, (Xi-Ni))               # [n_neighbors, n_neighbors]
		Si = Si + np.eye(n_neighbors)*tol*np.trace(Si)
		Si_inv = np.linalg.pinv(Si)
		wi = (np.dot(Si_inv, I)) / (np.dot(np.dot(I.T, Si_inv), I)[0,0])
		W[:, i] = wi[:,0]
	W_y = np.zeros((n, n))
	for i in range(n):
		index = N[i]
		for j in range(n_neighbors):
			W_y[index[j],i] = W[j,i]
	I_y = np.eye(n)
	M = np.dot((I_y - W_y), (I_y - W_y).T)
	eig_val, eig_vector = np.linalg.eig(M)
	index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
	Y = eig_vector[:, index_]
	return Y

if __name__ =='__main__':
    X,Y = make_s_curve(n_samples=500,noise=0.1,random_state=43)
    data = lle(X)







