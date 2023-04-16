import numpy as np
from DataSets import process_gpu_running_time
from sklearn.neural_network import MLPRegressor as MLPR
from PerformanceMetrics import MSE
from time import time

class MLPRegressor:
  def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='sgd',max_iter=200):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.act = activation
    self.solv = solver
    self.max_iter = max_iter

  def fit(self, x_train, y_train):
    #Create spaces for weights and biasses
    self.coef_ = [np.zeros((x_train.shape[1],self.hidden_layer_sizes[0]))]
    self.intercepts_ = [np.zeros((self.hidden_layer_sizes[0]))]
    for size in self.hidden_layer_sizes[1:]:
      self.coef_.append(np.zeros((self.coef_[-1].shape[1],size)))
      self.intercepts_.append(np.zeros((size)))
    self.coef_.append(np.zeros((self.coef_[-1].shape[1],1)))
    self.intercepts_.append(np.zeros((1)))

    #Calculate batch_size
    batch_size = min(200, x_train.shape[0])
    n_batches = y_train.shape[0]//batch_size

    for i_batch in range(n_batches):
      start_row = i_batch*batch_size
      sub_x = x_train[start_row:start_row+batch_size]
      sub_y = y_train[start_row:start_row+batch_size]
      for i in range(self.max_iter):
        #

  def predict(self, x_test):
    return

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gpu_running_time()

  print("My model:")
  t0 = time()
  model = MLPRegressor()
  model.fit(x_train,y_train)
  #print("MSE = ", MSE(y_test, model.predict(x_test)))
  #print("Elapse time = {:.5f}".format(time() - t0))

  #print("\nSKLEARN model:")
  #t0 = time()
  #model = MLPR()
  #model.fit(x_train,y_train)
  #Try to predict from made up data
  #print("MSE = ", MSE(y_test, model.predict(x_test)))
  #print("Elapse time = {:.5f}".format(time() - t0))

