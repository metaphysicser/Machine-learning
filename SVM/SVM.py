from sklearn import svm
import pandas as pd

x = [[2,0],[1,1],[2,3]]
y = [-1,-1,1]
clf = svm.SVC(kernel = 'linear')
clf.fit(x,y)#训练数据
print(clf.support_vectors_)#所有的支持向量
print(clf.support_)#所有支持向量的坐标
print(clf.n_support_)#两类支持向量的个数
print(clf.predict([[2,3],[0,0]]))#预测
