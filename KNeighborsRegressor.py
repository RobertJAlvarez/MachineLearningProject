import numpy as np
import pandas as pd
from PerformanceMetrics import MSE
from DataSets import process_gpu_running_time
from sklearn.neighbors import KNeighborsRegressor as KNN
from time import time

class KNeighborsRegressor:
  # Constructor
  def __init__(self,k=1,weighted = False):
    self.k = k
    self.weighted = weighted

  # Fit model parameters to training data
  def fit(self,x_train,y_train):
    self.x_train = x_train
    self.y_train = y_train
    self.y_unique = np.unique(y_train)

  # Predict class of test data
  def predict(self,x_test):
    # Improved version using the fact that (a-b)**2 = a**2 - 2ab + b**2
    dist1 = np.sum(x_test**2,axis=1,keepdims=True)
    dist2 = -2*np.matmul(x_test,self.x_train.T)
    dist3 = np.sum(self.x_train.T**2,axis=0,keepdims=True) # This part is not really necessary, since it does not depend on x_test
    dist = dist1+dist2+dist3

    # Distance matrix is created, get the k closest elements
    minIdxs = np.argpartition(dist, kth=self.k, axis=-1)[:,:self.k]
    return np.mean(self.y_train[minIdxs],axis=1)

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gpu_running_time()

  print("My model:")
  for k in range(1,16,2):
    print('k =',k)
    t0 = time()
    model = KNeighborsRegressor(k=k)
    model.fit(x_train,y_train)
    print("MSE = ", MSE(y_test, model.predict(x_test)))
    print("Elapse time = {:.5f}".format(time() - t0))

  print("\nSKLEARN model:")
  for k in range(1,16,2):
    print('k =',k)
    t0 = time()
    model = KNN(n_neighbors=k)
    model.fit(x_train,y_train)
    print("MSE = ", MSE(y_test, model.predict(x_test)))
    print("Elapse time = {:.5f}".format(time() - t0))

