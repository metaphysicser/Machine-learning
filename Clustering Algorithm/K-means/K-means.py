"""
@date:created on July 30th 21:06 2020
@author:张平路
@function:realize the K-means
"""
import numpy as np
import matplotlib as plt
from sklearn import datasets,model_selection

"""
@function:calculate the diatances
@parameter:x,y - two point
@return:dist - the distance between two points
"""
def distEclud(x,y):
    return  np.sqrt(np.sum((x-y)**2))

"""
@function:select k centroids
@parameter: data 
            k - the num of clusters
@return:centroid
"""
def ranCent(data,k):
    m,n = data.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m))
        centroids[i,:] = data[index,:]
    return centroids

"""
@function"ralize K-means
@parameter:data
            k
@return:
"""
def KMeans(data,k):
    m = np.shape(data)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    #first row store the type
    #second row store the distances to the cetroids
    clusterChange = True

    centroids = ranCent(data,k)
    print(centroids)
    while clusterChange:
        clusterChange =False

        for i in range(m):
            minDist = 100000.0
            minIndex = 0

            for j in range(k):
                distance = distEclud(centroids[j,:],data[i,:])
                if distance <minDist:
                    minDist = distance
                    minIndex = j

            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        print(clusterAssment)
        print(centroids)
        #update the centroids
        for j in range(k):
            temp = []
            for i in range(len(clusterAssment)):
                if clusterAssment[i,0] == j:
                    temp.append(i)
            centroids[j, :] = np.mean(temp, axis=0)
        return centroids,clusterAssment

def score(y_test,y_p):
    score = 0
    for i in range(len(y_test)):
        if y_test[i] == y_p[i]:
            score += 1
    return  float(score/len(y_test))



if __name__ =='__main__':
    Data = datasets.load_iris()
    #load the iris data

    data = Data['data']
    target = Data['target']
    print(target)
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data,target,random_state = 666)

    centroids,cluster = KMeans(X_train,3)
    y_p = np.array((cluster[:,0]).reshape((1,-1)))
    y_p = y_p[0].astype(int)

    print(score(y_p,y_train))