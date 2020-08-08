#-*- coding:UTF-8-*-
"""
created on Thurday June 16 19:24 2020
@功能：基于SVM算法的底层实现
@创建者：张平路
"""

from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import random
import types
import math

"""
@功能：读取文本数据
@参数：filename - 文件名
@返回值：dataMat - 数据矩阵
        labelMat - 数据标签
"""
def loaddataSet(filename):
    dataMate = []#矩阵
    labelMat = []#标签
    fr = open(filename)#打开文件
    for line in fr.readlines():#逐行读取
        lineArr = line.strip().split('\t')#去除空格并分离
        dataMate.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMate,labelMat

"""
@功能：随缘蹲一个j
@参数：i - 等缘人i
      m - 有缘人数目m
@返回：j - 有缘人
      
"""
def selectJrand(i,m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

"""
@功能：修剪alpha
@参数：L —— 下限
      H ——— 上限
      j ————要修剪的值
@返回：修剪后的alpha——j
      
"""
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L>aj:
        aj = L
    return aj


"""
@功能：简化版smo算法
@参数：dataMat - 数据矩阵
      labelMat - 数据标签
      C - 松弛变量
      toler - 容错率
      maxIter - 最大迭代次数
@返回： b - 截距
       alphas - 约束系数

"""
def simpleSMO(dataMat,labelMat,C,toler,maxIter):
    dataMatrix = np.mat(dataMat)#转换为矩阵
    labelMatrix = np.mat(labelMat).transpose()#将行向量变成列向量

    b = 0#初始化b
    m,n = np.shape(dataMatrix)

    alpha = np.mat(np.zeros((m,1)))#初始化alphas

    iter_num = 0#迭代次数

    while(iter_num<maxIter):
        alphaChanged = 0
        for i in range(m):
            #步骤1：计算误差
            fxi = float(np.multiply(alpha,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T)) +b
            Ei = fxi - float(labelMatrix[i])
            #下面这个判断语句真不会了，呜呜呜
            if ((labelMatrix[i] * Ei < -toler) and (alpha[i] < C)) or ((labelMatrix[i] * Ei > toler) and (alpha[i] > 0)):
                j = selectJrand(i,m)
                #计算误差Ej
                fxj = float(np.multiply(alpha,labelMatrix).T*(dataMatrix*dataMatrix[j,:].T)) +b
                Ej = fxj - float(labelMatrix[j])
                #保存下更新前的alphas
                alphaIold = alpha[i].copy()
                alphaJold = alpha[j].copy()
                #使用深拷贝

                #分情况讨论
                if(labelMatrix[i] != labelMatrix[j]):
                    L = max(0,alpha[j]-alpha[i])
                    H = min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0,alpha[j]+alpha[i]-C)
                    H = min(C,alpha[j]+alpha[i])
                if L==H: continue
                #步骤3，计算学习率eta
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:continue
                #步骤4：更新alpha——j
                alpha[j] -= labelMatrix[j]*(Ei - Ej)/eta
                #步骤6：修剪alpha—j
                alpha[j] = clipAlpha(alpha[j],H,L)
                if(abs(alpha[j]-alphaJold<0.00001)):continue
                #步骤6：更新alpha——i
                alpha[i] += labelMatrix[j]*labelMatrix[i]*(alphaJold - alpha[j])
                #步骤7：更新b_1和b_2
                b1 = b - Ei - labelMatrix[i] * (alpha[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMatrix[j] * (alpha[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMatrix[i] * (alpha[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMatrix[j] * (alpha[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alpha[i]) and (C > alpha[i]):
                    b = b1
                elif (0 < alpha[j]) and (C > alpha[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaChanged += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaChanged))
        if (alphaChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alpha


"""
@功能：计算w
@参数：dataMat - 数据矩阵
       labelMat - 数据标签
       alpha 
@返回：w的值 - 斜率

"""

def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)#-1表示任意列 重复行一次，列两次
    return w.tolist()

"""
@功能：数据可视化
@参数：dataMab - 数据矩阵
       w - 斜率
       b - 截距
@返回：无
"""
def showClassier(dataMat,w,b):
    data_plus = []#正样本
    data_minus =[]#负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
        data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
        data_minus_np = np.array(data_minus)
    #print(data_minus_np)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()
"""
@function:predict
@parament:
@return:predict
"""
def predict(p_dataMat,w,b):
    w1 = np.mat(w).transpose()
    y =np.dot(w1,p_dataMat.T) +b
    predict_y =np.sign(y)

    return predict_y

"""
@功能：主函数
@参数：无
@返回：无
"""
if __name__== '__main__':
    C = 0.1
    toler = 0.01
    dataMat,labelMat = loaddataSet('testSet.txt')

    b, alphas = simpleSMO(dataMat, labelMat, C, toler, 40)
    w = get_w(dataMat,labelMat,alphas)
    x =np.mat([[0,5],[6,-10]])
    y = predict(x,w,b)
    showClassier(dataMat, w, b)
    print(y)






