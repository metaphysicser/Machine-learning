# -*- coding:UTF8 -*-
"""
created on Monday 20th July 16:30 2020
@creator:张平路
@function:realize DecisionTree by using C45 algorithm and Pruning Algorithm

"""

from numpy import *
import numpy as np
import operator
from math import log
import json




"""
@function:find the most result
@parameter:classList - the result list
@return: the most result

"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #range the result by the down order
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

"""
@function:calculate the imformation entropy of data
@parameter: dataSet - the main data
@return: shannonEnt - the entropy of data
"""
def calcShannonEnt(dataSet):
    numEntropies = len(dataSet)

    labelCounts = {}#restore the number of different result
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
        labelCounts[currentLabel]+=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntropies)
        shannonEnt -= prob*log(prob,2)


    return shannonEnt

"""
@function:split the continous variable
@parameter:dataSet
           axis
           value
           direction - the direction to the split way
@return: retDataSet - a split data without selected feather
"""

def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                retDataSet.append(featVec[:axis] + featVec[axis + 1:])
        else:
            if featVec[axis] <= value:
                retDataSet.append(featVec[:axis] + featVec[axis + 1:])
    return retDataSet

"""
@function:split the discret variable
@parameter:dataSet
           axis
           value
           
@return: retDataSet - a split data without selected feather
"""
def splitDataSet(dataSet,axis,value):
    returnMat = []
    for data in dataSet:
        if data[axis]==value:
            returnMat.append(data[:axis]+data[axis+1:])
    return returnMat


"""
@function:choose the best split
@parameter:dataSet - the main data
           label - the label of data
@return: the best Feather - the feather of highest information gain
"""


def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = -1#只要把0改成-1就行，这个破bug修了我一晚上
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 对连续型特征进行处理 ,i代表第i个特征,featList是每次选取一个特征之后这个特征的所有样本对应的数据
        featList = [example[i] for example in dataSet]
        # 因为特征分为连续值和离散值特征，对这两种特征需要分开进行处理。
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEntropy = 10000
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for value in splitList:
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)

                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = value
                    # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = bestSplit
            infoGain = baseEntropy - bestSplitEntropy

        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵,选取第i个特征的值为value的子集
            for value in uniqueVals:

                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue,例如将密度变为密度<=0.3815
    # 将属性变了之后，之前的那些float型的值也要相应变为0和1

    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':

        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        #print(labels[bestFeature])
        for i in range(shape(dataSet)[0]):
           if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 0
                #print(dataSet[i][bestFeature])

           else:
               dataSet[i][bestFeature] = 1
    return bestFeature




"""
@function:create a decisionTree
@parameter:dataSet - the current data
           label - the current label
           data_full - the whole data
           label_full - the whole label
@return: a built decisionTree
"""


def createTree(dataSet, labels, data_full, labels_full):

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # none feather
        return classList[0]

    if len(dataSet[0]) == 1:  # only one feather
        return majorityCnt(classList)
    # 平凡情况，每次找到最佳划分的特征

    bestFeat = chooseBestFeatureToSplit(dataSet, labels)

    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}
    #print("bestfeat",bestFeat)

    featValues = [example[bestFeat] for example in dataSet]


    '''
    刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块
    ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了，
    事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。
    如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。
    但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。
    '''
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])
        #print("data_full",data_full)

        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])#delete the best feather from label


    '''
    针对bestFeat的每个取值，划分出一个子树。对于纹理，树应该是{"纹理"：{？}}，显然？处是纹理的不同取值，有清晰模糊和稍糊三种，对于每一种情况，
    都去建立一个自己的树，大概长这样{"纹理"：{"模糊"：{0},"稍糊"：{1},"清晰":{2}}}，对于0\1\2这三棵树，每次建树的训练样本都是值为value特征数减少1
    的子集。
    '''


    for value in uniqueVals:
        subLabels = labels[:]

        data_temp = []
        for data in dataSet:#delete the data of best feather from the whole data
            if data[bestFeat] == value:
              data_temp.append(data[:bestFeat] + data[bestFeat + 1:])


        if type(dataSet[0][bestFeat]).__name__ == 'str':
            #print('bestfeather',bestFeatLabel)
            #print('value',value)
            #print('uniquevals',uniqueVals)
            #print('uniquevalsFull',uniqueValsFull)

            uniqueValsFull.remove(value)
            #print('after uniquevals', uniqueVals)
            #print('after uniquevalsFull', uniqueValsFull)


        myTree[bestFeatLabel][value] = createTree(data_temp, subLabels, data_full, labels_full)


    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree


"""
@function: predict the test data
@parameter:mytree - The decisionTree
           x_test - teh feather
           y_test - the result
           label
@return: res - predict result
         score - the righr ratio
        
"""

def predict(x,y_test,label,key,value):
    total = shape(x)[0]
    res = []

    #print(x)
    #print(value)
    #key = list(mytree.keys())[0]
    #value = list(mytree.values())[0]
    if type(value).__name__ == 'str':
        return value


        #print(label)


    #print(key)
    #key = key[0].strip().split('<=')
    if "<=" in key:

       key = key.strip().split('<=')
       key_index = label.index((key[0]))
       key[1] = float(key[1])
       #print(type(x[key_index]))
       if x[key_index] < key[1]:
           x[key_index] = 0
       else:
           x[key_index] = 1
    else:
        key_index = label.index(key)




    key_data = x[key_index]
    #print(key_data)
    if type(value[key_data]).__name__ =='str':
        key_2 = list(value[key_data])[0]
    else:

        key_2 = list(value[key_data].keys())[0]

    if type(value[key_data]).__name__ =='str':
        value_2 = list(value[key_data])[0]
    else:
        value_2 = list(value[key_data].values())[0]
    #print(key)
    #print(value_2)
    #mytree_1 = {key_2:value_2}

    res.append(predict(x,y_test,label,key_2,value_2))





    return res
"""
@function:main function
@parameter:nome
@return:none
"""

if __name__ =='__main__':
    fr = open('melon.txt')
    Dataset = [i.strip().split(',') for i in fr.readlines()]#creat the set of data

    label = Dataset[0][1:-1]
    label_full = Dataset[0][1:-1]
    #create the label of data
    dataset = [row[1:] for row in Dataset[1:]]
    dataset_full = [row[1:] for row in Dataset[1:]]
    #gain the main data

    for i in range(len(dataset)):#transform the string into int
        dataset[i][7] = float(dataset[i][7])
        dataset[i][6] = float(dataset[i][6])
        dataset_full[i][7] = float(dataset[i][7])
        dataset_full[i][6] = float(dataset[i][6])
    #print(label.index('纹理'))
    #print(type(dataset))

    mytree  = createTree(dataset,label,dataset_full,label_full)

    #f = {e[0]:d[0]}
    #print(label_full)
    key = list(mytree.keys())[0]
    value = list(mytree.values())[0]
    predict1 = []
    for i in range(shape(dataset_full)[0]):
        predict1.append(predict(dataset_full[i][:-1],dataset_full[i][-1],label_full,key,value)[0][0])
    #print(predict1)
    right = 0
    for i in range(len(predict1)):
        if predict1[i] == dataset_full[i][-1]:
            right +=1

    print("精度为",float(right/len(predict1)))


    print(json.dumps(mytree, ensure_ascii=False, indent=4))





