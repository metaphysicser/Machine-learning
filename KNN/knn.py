"""
@date:created on July 23 19:03 2020
@author:张平路
@function:realize the knn

"""
import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


"""
@function:the gaussian function
@parameter: dis - the independent variable distance
@return: weight - the dependent variable weight
"""


def gaussian(dist, sigma=10.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


"""
@function:realize the knn
@parameter:
@return:X_train - the train data
        X_test - the test data
        y_train - the train target
        y_test - the test target
@return:score - the right score

"""
def knn(X_train,X_test,y_train,y_test):
    k = 11 #超参数取11

    predict_true = 0 #the num of right predicted
    max = len(y_test)#the max num of iteration

    for i in range(max):
        x_p = X_test[i]
        y_p = y_test[i]

        distances = [np.sqrt(np.sum((x_p - x) ** 2)) for x in X_train]
        #calculate the distance between point in x_p and point in x
        d = np.sort(distances)
        #sort the distances
        nearest = np.argsort(distances)
        #the index of sorted data
        #print(nearest)

        topk_y = [y_train[j] for j in nearest[:k]]
        #select k nearest num

        classCount = {}
        for i in range(0, k):
            voteLabel = topk_y[i]
            weight = gaussian(distances[nearest[i]])
            # print(index, dist[index],weight)
            ## 这里不再是加一，而是权重*1
            classCount[voteLabel] = classCount.get(voteLabel, 0) + weight * 1

        maxCount = 0

        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                classes = key
        #select the type of max num
        if (classes == y_p): predict_true += 1

        precision = predict_true / max

    return precision
"""
@function: the access to the main program
@parameter:none
@return:none
"""
if __name__ =='__main__':
    data = load_iris()
    X_train,X_test,y_train,y_test =train_test_split(data['data'],data['target'],random_state=0)

    score = knn(X_train,X_test,y_train,y_test)
    print((score))