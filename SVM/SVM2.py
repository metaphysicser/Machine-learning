
# -*- coding: utf-8 -*-
"""
Created on Thurday July 16 14:38 2020
@功能：调用库函数的支持向量机的简单应用
@author: 张平路
"""
import numpy as np
import pylab as pl
from sklearn import  svm
"""
@功能：获得训练样本
@参数：无
@返回：20个二维矩阵
"""
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
#利用获得的数据进行训练
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

#计算直线的参数
w = clf.coef_[0]
a  = -w[0]/w[1]#获得斜率
xx = np.linspace(-5,5)
yy = a*xx-(clf.intercept_[0]/w[1])#intercept 为截距
#找到两条边界直线
b = clf.support_vectors_[0]
yy_down = a* xx +(b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up = a* xx +(b[1] - a*b[0])  # 两条虚线

print("w: ", w)
print("a: ", a)
#print(" xx: ", xx)
#print(" yy: ", yy)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

pl.plot(xx, yy, 'k-')#最优直线
pl.plot(xx, yy_down, 'k--')#边界直线
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
