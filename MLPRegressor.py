import numpy as np
from DataSets import process_gpu_running_time
from sklearn.neural_network import MLPRegressor as MLPR
from PerformanceMetrics import MSE
from Utils import relu
from time import time

class MLPRegressor:
  def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.001):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.act = activation
    self.solv = solver
    self.lern_rate = learning_rate_init

  def fit(self, x_train, y_train):
    #Create spaces for weights and biasses
    self.coef_ = [np.random.randn(x_train.shape[1],self.hidden_layer_sizes[0])*np.sqrt(2./self.hidden_layer_sizes[0])]
    self.intercepts_ = [np.zeros((self.hidden_layer_sizes[0]))]
    for size in self.hidden_layer_sizes[1:]:
      self.coef_.append(np.random.randn(self.coef_[-1].shape[1],size)*np.sqrt(2./self.hidden_layer_sizes[0]))
      self.intercepts_.append(np.zeros((size)))
    self.coef_.append(np.random.randn(self.coef_[-1].shape[1],1)*np.sqrt(2.))
    self.intercepts_.append(np.zeros((1)))

    #Calculate batch_size
    batch_size = min(200, x_train.shape[0])
    n_batches = y_train.shape[0]//batch_size

    for i_batch in range(n_batches):
      Z = [0]*len(self.coef_)
      A = [0]*len(self.coef_)
      dZ = [0]*len(self.coef_)
      dW = [0]*len(self.coef_)
      db = [0]*len(self.coef_)

      #Subset x_train and y_train for this batch
      start_row = i_batch*batch_size
      sub_x = x_train[start_row:start_row+batch_size]
      sub_y = y_train[start_row:start_row+batch_size].reshape((-1,1))

      #Forward propagation
      Z[0] = np.matmul(sub_x,self.coef_[0]) + self.intercepts_[0]
      A[0] = relu(Z[0])
      for i in range(1,len(Z)):
        Z[i] = np.matmul(A[i-1],self.coef_[i]) + self.intercepts_[i]
        A[i] = relu(Z[i])

      #Backward propagation
      i = len(Z)-1
      dZ[i] = A[i] - sub_y
      dW[i] = np.matmul(A[i-1].T,dZ[i],)/batch_size
      db[i] = np.sum(dZ[i],axis=1,keepdims=True)/batch_size
      for i in range(len(Z)-2,0,-1):
        dZ[i] = np.matmul(dW[i+1].T,dZ[i+1]) * relu(Z[i])
        dW[i] = np.matmul(dZ[i],A[i].T)/batch_size
        db[i] = np.sum(dZ[i],axis=1,keepdims=True)/batch_size
      dZ[0] = np.matmul(dZ[1],dW[1].T) * relu(Z[0])
      dW[0] = np.matmul(sub_x.T,dZ[0])/batch_size
      db = np.sum(dZ[0],axis=1,keepdims=True)/batch_size

      #Update weights and biases
      for i in range(len(Z)):
        self.coef_[i] -= self.lern_rate*dW[i]
        self.intercepts_[i] -= self.lern_rate*db[i]

    #for i in range(len(self.coef_)):
    #  print("self.coef_[i].shape = ", self.coef_[i].shape)
    #  print("self.intercepts_[i].shape = ", self.intercepts_[i].shape)

  def predict(self, x_test):
    H = np.matmul(x_test,self.coef_[0]) + self.intercepts_[0]
    for i in range(1,len(self.coef_)):
      H = relu(H)
      H = np.matmul(H,self.coef_[i]) + self.intercepts_[i]
    return H

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gpu_running_time()

  print("My model:")
  t0 = time()
  model = MLPRegressor()
  model.fit(x_train,y_train)
  print("MSE = ", MSE(y_test, model.predict(x_test)))
  print("Elapse time = {:.5f}".format(time() - t0))

  #print("\nSKLEARN model:")
  #t0 = time()
  #model = MLPR()
  #model.fit(x_train,y_train)
  #Try to predict from made up data
  #print("MSE = ", MSE(y_test, model.predict(x_test)))
  #print("Elapse time = {:.5f}".format(time() - t0))

