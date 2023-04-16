import numpy as np
from PerformanceMetrics import accuracy
from DataSets import process_gamma_dataset, mnist
from Utils import to_categorical
from sklearn.linear_model import LogisticRegression as LR
from time import time

def standardize(x):
  for i in range(x.shape[1]):
    x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])

class LogisticRegression:
  def P_(self,X,B):
    N = X.shape[0]  #Number of object in training set
    P = X.shape[1]
    a = np.empty((N,self.K))
    for i in range(self.K):
      a[:,i] = np.reshape(np.exp(np.dot(X,B[i*P:i*P+P])),(-1))
    a /= np.sum(a,axis=1).reshape((-1,1))
    return np.reshape(a,(-1,1))

  def __init__(self, penalty=None, tol=1e-4, max_iter=100):
    self.penalty = penalty
    self.tol = tol
    self.max_iter = max_iter

  def fit(self, x_train, y_train, alpha=0.001):
    self.K = np.unique(y_train).shape[0] #Number of classes
    K = self.K
    if K == 2:
      print("This is a binary logistic regression")
    #Add a column of 1's at the beginning of the data to calculate y intercept
    X = np.c_[np.ones((x_train.shape[0],1)),x_train]
    N = X.shape[0]  #Number of object in training set
    P = X.shape[1]  #Number of parameters + 1 because of the 1's column
    #Reshape y_train so it is a 1 column matrix
    Y = to_categorical(y_train).reshape((-1,1))
    #Set beta values to 0
    B = np.zeros(shape=(K*P,1))
    new_B = np.empty(shape=(K*P,1))
    for _ in range(self.max_iter):
      p = self.P_(X,B)
      y_p = Y-p
      for i in range(K):
        W = p[i*N:i*N+N]*(1-p[i*N:i*N+N])
        XTWX = np.matmul(X.T,W*X)
        temp = np.matmul(np.linalg.pinv(XTWX),X.T)
        new_B[i*P:i*P+P] = B[i*P:i*P+P] - np.matmul(temp,y_p[i*N:i*N+N])
      if np.sum(abs(new_B-B)) < self.tol: break
      B = new_B
    self.intercept_ = np.array(B[0::P])
    #Get all betas that are not the intercepts and store it as a 2D array, 1 row per class coefficients
    self.coef_ = np.delete(B,list(range(0,B.shape[0],P)),axis=0).reshape((-1,P-1))

  def predict(self, x_test):
    X = np.c_[np.ones((x_test.shape[0],1)),x_test]
    B = np.hstack((self.intercept_,self.coef_)).reshape((-1))
    p = np.reshape(self.P_(X,B),(self.K,-1))
    return np.argmin(p,axis=0)

  def __str__(self):
    return '{}(max_iter={:03}, penalty={})'.format(self.__class__.__name__,self.max_iter,self.penalty)

if __name__ == '__main__':
  #x_train,x_test,y_train,y_test = process_gamma_dataset()
  x_train,x_test,y_train,y_test = mnist()

  #standardize(x_train)
  #standardize(x_test)

  #My model
  print("My model")
  t0 = time()
  model = LogisticRegression()
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.3f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

  #Run sklearn simple model
  print("\nSKLEARN model")
  t0 = time()
  model = LR(max_iter=100, penalty=None, tol=1e-4)
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

