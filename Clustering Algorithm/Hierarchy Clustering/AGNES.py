"""
@date:2020 8.5 9:49
@author:张平路
@function:realize Hierarchy Clustering
"""
import math
import pylab as pl
import numpy as np


"""
@function:calculate the diastance between a and b
@parameter:a
           b
@return:distance
"""
def dist(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2

    return math.sqrt(sum)

"""
@function:find the minor index
@parameter:data_dist
@return:x,y,min

"""
def find_min(data_dist):
    min = 10000
    x =0
    y =0
    for i in range(len(data_dist)):
        for j in range(len(data_dist[i])):
            if i != j and data_dist[i][j] <min:
                min = data_dist[i][j]
                x = i
                y = j

    return (x,y,min)
"""
@function:intrgrate near clusters
@parameter:data
           dist - the distance between two chusters
           k - the final num of clusters
return:
"""
def AGNES(data_list,dist,k):

    data_dist = []



    for i in data_list:
        temp = []
        for j in data_list:
            temp.append(dist(i,j))
        data_dist.append(temp)

    m = len(data_list)
    cluster = []
    for i in range(m):
        cluster.append([i])
    count = 1
    #integrate near clusters
    while m>k:
        x,y,min = find_min(data_dist)

        data_list[x] = [(data_list[x][i]+data_list[y][i])/2 for i in range(len(data[x]))]
        data_list.remove(data_list[y])

        cluster[x] = cluster[x]+cluster[y]
        cluster.remove(cluster[y])

        data_dist = []

        for i in data_list:
            temp = []
            for j in data_list:
                temp.append(dist(i, j))
            data_dist.append(temp)


        m -= 1
        print('合并完成第' + str(count) + '次,这时簇的个数为' + str(m))
        count = count+1


    return cluster


def draw(cluster,data):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            pl.scatter(data[cluster[i][j]][0],data[cluster[i][j]][1],color = colValue[i%len(colValue)])
    pl.show()




if __name__ =='__main__':

    fr = open('points.txt')
    data = []
    for line in fr.readlines():

        temp = line.strip().split(',')
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        data.append(temp)
    data_copy = data.copy()






    cluster = AGNES(data_copy,dist,7)
    print(cluster)
    draw(cluster,data)