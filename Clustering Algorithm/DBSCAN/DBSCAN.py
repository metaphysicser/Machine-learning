"""
@date:2020 8.4 18:14
@author:张平路
@function:realize the DBSCAN algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import math

unvisited = -2  # the flag of point that has not been visited
noise = -1
"""
伪代码：
初始化参数
坐标样本集： D (x1, x2, ...... , xm)        （这里是在网上找的一个很多点的txt文本）
邻域参数：  ϵ (相当于超球体的半径)    MinPts  (最小个数)
距离度量方式： 欧式距离
 
初始化所有的点为unvisited
while 所有unvisited样本:
    有unvisited样本p;
    将p记为visited;
    # 其实这里只是在计算核心对象，如果计算完不满足核心对象的话直接为噪声，不考虑非核心对象，因为簇由核心对象决定
    if p的ϵ-邻域里有MinPts个其他样本:
        创建新簇C，将p添加到C;
        设置N为p的ϵ-邻域里的所有对象的集合
        # 遍历N中的所有的点
        for N中每个点 p':
            if p'是unvisited:
                将p'记为visited;
                if p'的ϵ-邻域里有MinPts个其他样本:
                    将这些样本添加到N;
                if p'不是任何簇的成员:
                    将p'添加到C
            # 这一步是自己新添的
            if p'是噪声:
                抹去噪声标记并添加到C
    else 标记p为噪声;


"""
"""
@function:find the point in circle
@parameter:data
           SelectedId 
           radius - the radius of circle
@return:the points in domain
"""
def whether_kernal_object(data,selectedId,radius):
    neighbor = []
    for i in range(np.shape(data)[0]):
        if math.sqrt(np.power(data[selectedId] - data[i],2).sum()) < radius:
            neighbor.append(i)

    return neighbor






"""
@function:find the cluster
@parameter:data
@return:none
"""
def DBSCAN(data,radius,minPts):
    pointNum = np.shape(data)[0]  # the total num of points
    flagClusterList = [unvisited] * pointNum
    clusterNumber = 1  # the index of cluster

    for selectPoint in range(pointNum):
        if flagClusterList[selectPoint] == unvisited:
            neighbors = whether_kernal_object(data,selectPoint,radius)
            if len(neighbors) < minPts:
                flagClusterList[selectPoint] = noise#the cluster is a noise
            else:
                flagClusterList[selectPoint] = clusterNumber
                #pack other all points in the cluster
                for neighborsId in neighbors:
                    if flagClusterList[neighborsId] == unvisited or flagClusterList[neighborsId] == noise:
                        flagClusterList[neighborsId] = clusterNumber

                while len(neighbors):#expand teh current cluster
                    currentPoint = neighbors[0]
                    queryNeighbors = whether_kernal_object(data,currentPoint,radius)

                    if len(queryNeighbors) >= minPts:#include the points in new circle
                        for i in range(len(queryNeighbors)):
                            resultPoint = queryNeighbors[i]
                            if flagClusterList[resultPoint] == unvisited:
                                neighbors.append(resultPoint)
                                flagClusterList[resultPoint] = clusterNumber

                            elif flagClusterList[resultPoint] == noise:
                                flagClusterList[resultPoint] = clusterNumber
                    neighbors = neighbors[1:]
                clusterNumber += 1
    print("簇的个数"+str(clusterNumber-1))
    visualization(data,flagClusterList,clusterNumber-1)



def visualization(dataSet, resultSet, clusterNumber):
    """
    实现可视化
    输入：数据集, 结果集, 簇个数
    """
    # 转一下
    matResult = np.mat(resultSet).transpose()
    dataSet = dataSet.transpose()
    # 图
    figure = plt.figure()
    # 取红橙黄绿蓝几种颜色
    colors = ['green', 'blue', 'orange', 'yellow', 'red']
    # 一行一列第一个
    ax = figure.add_subplot(111)
    # 遍历每一个簇
    for i in range(clusterNumber + 1):
        # 选一种颜色，若是簇的数目超过颜色数的话从头选
        color = colors[i % len(colors)]
        sub = dataSet[:, np.nonzero(matResult[:, 0].A == i)]
        # 画点 前两个参数为坐标
        ax.scatter(sub[0, :].flatten().A[0], sub[1, :].flatten().A[0], c=color, s=40)
    plt.show()





if __name__ =='__main__':
    radius = 2
    minPts = 15#the number of points in circle
    unvisited = -2 #the flag of point that has not been visited
    noise = -1


    fr = open('points.txt')
    data = []
    for line in fr.readlines():

        temp = line.strip().split(',')
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        data.append(temp)
    data = np.mat(data)

    DBSCAN(data,radius,minPts)

