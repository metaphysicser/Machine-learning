#-*-coding:UTF-8-*-
"""
@date:2020 8.6 14:30
@author:张平路
@function:realize Isomap Algorithm
"""
import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D


"""
@function:construt a new distance matrix using floyd
@parameter:D - the old distance matrix
           n_neighbors
@return:D1
"""
def floyd(D,n_neighbors):
    Max = np.max(D) * 1000
    n1, n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1, n1)) * Max
    D_arg = np.argsort(D, axis=1)  # 返回从小到大的索引值，每一列进行排序

    for i in range(n1):
        D1[i, D_arg[i, 0:k + 1]] = D[i, D_arg[i, 0:k + 1]]  # 找出与i最近的k个数
    print("最近的k个数已经找到")
    n = 1
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i, k] + D1[k, j] < D1[i, j]:
                    D1[i, j] = D1[i, k] + D1[k, j]  # 需再次理解
                    print("检查距离矩阵第"+str(n)+"次,共"+str(n1**3)+"次")

                    n = n+1
    return D1


def my_mds(dist, n_dims):    #距离矩阵分解，得到降维之后的数据
    # dist (n_samples, n_samples)
    dist = dist**2
    n = dist.shape[0]
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1)/n
    T3 = np.sum(dist, axis = 0)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]

    return picked_eig_vector*picked_eig_val**(0.5)



"""
@function:calculate the distance between two points
@parameter:data
@return:dist
"""
def cal_pairwise_dist(data):
    sum_x = np.sum(np.square(data),1)
    dist = np.add(np.add(-2 * np.dot(data, data.T), sum_x).T, sum_x)
    return dist


"""
@function:the Isomap Algorithm
@parameter:data
           n - the main demension in MDS
           n_neighbors - the num of points near center
@return:the data with lower dimension
"""
def my_Isomap(data,n = 2,n_neighbors = 30):
    D = cal_pairwise_dist(data)
    print("各点距离已经计算完成")
    D[D<0] = 0
    D = D**0.5
    D_floyd = floyd(D,n_neighbors)
    print("floyd算法计算成功")
    print("正在利用MDS算法降维")
    data_n = my_mds(D_floyd,n)
    print("降维成功")


    return data_n

def scatter_3d(X, y):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    plt.show()

if __name__ =='__main__':
    X,Y = make_s_curve(n_samples=500,noise=0.1,random_state=43)

    data_1 = my_Isomap(X, 2, 10)#用全部数据的话会计算一亿次，可以适当削减数据量

    data_2 = Isomap(n_neighbors=10, n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("my_Isomap")
    plt.scatter(data_1[:, 0], data_1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_Isomap")
    plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)
    plt.savefig("Isomap1.png")
    plt.show()
