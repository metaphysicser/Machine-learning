#-*-coding:UTF8 -*-
"""
created on Friday July 17 9:00 2020
@creator:张平路
@function:realizing the nonlinear SVM

"""

import matplotlib.pyplot as plt
import numpy as np
import random

"""
@function:transform data to a higher dimentional space by kernael function
@parameters:X - 数据矩阵
            A - 单个数据向量
            KTup - 包含核函数信息的元组, 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
@return: K - 计算的核K
"""
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X*A.T#linear kernal function,only inner product of X and A
    elif kTup[0] == 'rbf':#gauss function
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T #compute gauss kernal K
        K = np.exp(K/-1*kTup[1]**2)
    else:raise NameError('can not indentify kernal function')
    return K


"""
@function:gather every parameters
@parameters:
           dataMatIn - 数据矩阵
           classLabels - 数据标签
           C - 松弛变量
           toler - 容错率
           kTup - 包含核函数信息的元组
return:none
"""
class optStruction:
  def __init__(self,dataMatIn,classLabels,C,toler,kTup):
    self.X = dataMatIn
    self.labelMat = classLabels
    self.C  =C
    self.tol = toler
    self.m = np.shape(dataMatIn)[0]
    self.alphas = np.mat(np.zeros((self.m,1)))
    self.b = 0
    self.eCache = np.mat(np.zeros((self.m, 2)))  # 根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
    self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化核K
    for i in range(self.m):  # 计算所有数据的核K
        self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)#linear kernal function,only inner product of X and A

"""
@function: read data
@parament:filename - 文件名
@return: dataMat - 数据矩阵
         labelMat - 数据标签
"""

def loadData(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


"""
@function:compute inaccuracy
@parament:os - 数据结构
          k - 标号为k的数据
@return:Ek - the inarruracy of num k
"""
def calcEk(os,k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T * os.K[:, k] + os.b)
    Ek = fXk - float(os.labelMat[k])
    return Ek

"""
@function:choose alpha_y randomly
@parament:i - index
          m - the num of alpha
@return:j - index
"""

def selectJrand(i, m):

    j = i                                 #选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


"""
@function:内循环启发挑选j
@parament: i - index
           os - struture of data
           Ei - inarruracy of index i
"""
def selectJ(i,os,Ei):
    maxK = -1
    max = DeltaE = 0
    Ej = 0
    os.eCache[i] = [i,Ei]
    validEcacheList = np.nonzero(os.eCache[:, 0].A)[0]#返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i: continue  # 不计算i,浪费时间
            Ek = calcEk(os, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            maxDeltaE = 0
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, os.m)  # 随机选择alpha_j的索引值
        Ej = calcEk(os, j)  # 计算Ej
    return j, Ej
"""
@function:compute inaccuracy 
@parament: os - data structure
           k - index
@return:none
"""
def updateEk(oS, k):

    Ek = calcEk(oS, k)                                        #计算Ek
    oS.eCache[k] = [1,Ek]                                    #更新误差缓存
"""
@function:change alpha
@parament:aj - alpha_j的值
        H - alpha上限
        L - alpha下限
@return: alpha
"""
def clipAlpha(aj,H,L):

    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
@function:improved SMO
@parament: i - grade
           os - data struture
@return:1 - a couple of alpha have changed
        0 - a couple of alpha have not changed or changed too small
"""
def innerL(i,os):
    #step 1:compute inaccuracy Ei
    Ei = calcEk(os,i)
    if ((os.labelMat[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i, os, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = os.alphas[i].copy();
        alphaJold = os.alphas[j].copy();
        # 步骤2：计算上下界L和H
        if (os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        os.alphas[j] -= os.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(os, j)
        if (abs(os.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        os.alphas[i] += os.labelMat[j] * os.labelMat[i] * (alphaJold - os.alphas[j])
        # 更新Ei至误差缓存
        updateEk(os, i)
        # 步骤7：更新b_1和b_2
        b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.K[i, i] - os.labelMat[j] * (
                    os.alphas[j] - alphaJold) * os.K[i, j]
        b2 = os.b - Ej - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.K[i, j] - os.labelMat[j] * (
                    os.alphas[j] - alphaJold) * os.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0



"""
@function:complete SMO
@parament:dataMatIn - 数据矩阵
          classLabels - 数据标签
          C- 松弛变量
          toler - 容错率
          maxIter - 最大迭代次数
          kTup - 包含核函数信息的元组
@return: os.b - 截距
         os.alphas - 约束系数
"""
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)):
    os= optStruction(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        #alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)  # 使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, os)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return os.b, os.alphas


"""
@function: visual data
@parament: dataMat - data matrix
           labelMat - label of data
@return:none
"""

def showDataSet(dataMat, labelMat):

    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

"""
@function:predict
@parament:dataMat - 训练数据矩阵
          p_dataMat - 测试数据矩阵
          b - 截距
          alpha
@return:predict1 - 预测结果

"""
def predict1(dataMat,p_dataMat,b,alphas):
    k1 =1.3
    predict1 = []

    m, n = np.shape(p_dataMat)
    svInd = np.nonzero(alphas.A > 0)[0]  # 获得支持向量
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd];
    for i in range(m):
        kernelEval = kernelTrans(sVs, p_dataMat[i, :], ('rbf', k1))  # 计算各个点的核
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b  # 根据支持向量的点，计算超平面，返回预测结果
        #if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1  # 返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
        print(np.sign(predict))
        if float(predict)>= 0 : predict1.append(1)
        else:predict1.append(-1)

    print(predict1)
    return(predict1)

"""
@function:main function
@parament:none
@return:none

"""

def testRbf(k1 = 1.3):
    """
    测试函数
    Parameters:
        k1 - 使用高斯核函数的时候表示到达率
    Returns:
        无
    """
    dataArr,labelArr = loadData('testSetRBF.txt')                        #加载训练集
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))        #根据训练集计算b和alphas
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]                                        #获得支持向量
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))                #计算各个点的核
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b     #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1        #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("训练集错误率: %.2f%%" % ((float(errorCount)/m)*100))             #打印错误率
    dataArr,labelArr = loadData('testSetRBF2.txt')                         #加载测试集
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))                 #计算各个点的核
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b         #根据支持向量的点，计算超平面，返回预测结果
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1        #返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("测试集错误率: %.2f%%" % ((float(errorCount)/m)*100))             #打印错误率

if __name__ == '__main__':
    k1 =1.3
    dataArr, labelArr = loadData('testSetRBF.txt')  # 加载训练集
    d = np.array(dataArr)

    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))  # 根据训练集计算b和alphas
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()

    dataArr, labelArr = loadData('testSetRBF2.txt')  # 加载测试集

    p_datMat = np.mat(dataArr);
    p = predict1(datMat,p_datMat,b,alphas)
    int_label = []
    for i in labelArr:
        i = int(i)
        int_label.append(i)

    print(len(int_label))
    print(len(p))
    right = 0
    for i in range(len(p)):
        if p[i] == labelArr[i]:
            right +=1

    print("测试集的精度为",float(right/len(p)))






