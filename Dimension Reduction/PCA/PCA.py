#-*-coding:UTF8-*-
"""
@data:created on July 29th 15:24 2020
@author:张平路
@function；realize the PCA alogrithm

"""
import numpy as np
import pandas as pd
from sklearn import datasets

class DimensionValueError(ValueError):
    """定义异常类"""
    pass
class PCA(object):
    #define the PCA class
    def __init__(self,x,n_components = None):
        self.x = x
        self.dimension = x.shape[0]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")
        self.n_components= n_components

    """
    @function:calculate the covariance
    @parameter:self
    @return: x_cov 
    """
    def cov(self):
        x_T = np.transpose(self.x)
        x_cov = np.cov(x_T)
        return x_cov

    """
    @function:calculate the feature vertor and feature value
    @parameter:self
    @return:c_df_sort
    """
    def get_feature(self):
        x_cov = self.cov()
        a,b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)),b))
        c_df = pd.DataFrame(c)

        c_df_sort = c_df.sort_values(by = 0,ascending = False)

        return  c_df_sort





    """
    @function:decrese the data dimension by choosing dimension and variance
    @parameter:self
    @return: the decresed dimension data
    
    """
    def reduce_dimension(self):

        c_df_sort = self.get_feature()
        variance = c_df_sort.values[:,0]

        if self.n_components:
            p = c_df_sort.sort.values[0:self.n_components,1:]
            y = np.dot(p,np.transpose(self.x))
            return np.transpose(y)

        variance_sum = sum(variance)
        variance_radio = variance/variance_sum

        variance_contribution = 0
        for R in range(self.dimension):
            variance_contribution +=variance_radio[R]
            if variance_contribution > 0.99:
                break

        p = c_df_sort.values[0:R+1,1:]
        y = np.dot(p,np.transpose(self.x))
        return np.transpose(y)

digits = datasets.load_digits()
x = digits.data

if __name__ == '__main__':
    pca = PCA(x)
    y = pca.reduce_dimension()


