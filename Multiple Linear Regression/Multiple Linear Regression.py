import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from  math import sqrt
class LinearRegression(object):
      def __init__(self):
          # coef相当于简单线性回归中的a
          self.coef_=None
          # interception相当于简单线性回归中的b
          self.interception_=None
          self._theta=None
      #训练模型
      def fit_normal(self,X_train,y_train):
          assert X_train.shape[0]==y_train.shape[0],'params errors!'
          X_b=np.hstack([np.ones((len(X_train),1)),X_train])#增广矩阵
          self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)#计算theta
          self.coef_=self._theta[1:]
          self.interception_=self._theta[0]
          return self

      def predict(self,X_predict):
          assert self.coef_ is not None ,'fit before predict'
          assert X_predict.shape[1]==len(self.coef_),'params errors'
          X_predict=np.hstack([np.ones((len(X_predict),1)),X_predict])
          y_predict=X_predict.dot(self._theta)
          return y_predict

      def score(self,X_test,y_test):#R方
          y_predict=self.predict(X_test)
          return Rsquare(y_test,y_predict)
def mean_square_error(y_true,y_predict):
    y_true=np.array(y_true)
    y_predict=np.array(y_predict)
    assert len(y_true)==len(y_predict),'the length must be same'
    return np.sum((y_true - y_predict) ** 2) / len(y_predict)
def root_mean_square_error(y_true,y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    assert len(y_true) == len(y_predict), 'the length must be same'
    return sqrt(np.sum((y_true-y_predict)**2)/len(y_predict))

def mean_abs_error(y_true,y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    assert len(y_true) == len(y_predict), 'the length must be same'
    return np.sum(np.abs(y_true-y_predict))/len(y_predict)
def Rsquare(y_true,y_predict):
    return 1-mean_square_error(y_true,y_predict)/np.var(y_true)
boston=datasets.load_boston()
x=boston.data
y=boston.target
x=x[y<50.0]
y=y[y<50.0]
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=666)
linear=LinearRegression()
linear.fit_normal(X_train,y_train)
print(linear.coef_)
print(linear.interception_)
score=linear.score(X_test,y_test)
print(score)

