import numpy as np
from PerformanceMetrics import accuracy
from DataSets import process_gamma_dataset, mnist
from Utils import to_categorical, standardize
from sklearn.linear_model import LogisticRegression as LR
from time import time
from NPTypes import NumNPArrayNxM, ArrayLike, NumNPArray

class LogisticRegression:
  def __P(self, X: NumNPArrayNxM, B: NumNPArray) -> NumNPArray:
    N = X.shape[0]  #Number of object in training set
    P = X.shape[1]
    a = np.empty((N,self.K))
    for i in range(self.K):
      a[:,i] = np.reshape(np.exp(np.dot(X,B[i*P:i*P+P])),(-1))
    a /= np.sum(a,axis=1).reshape((-1,1))
    return np.reshape(a,(-1,1))
    #return np.reshape(a,(-1,1),order='F')

  def __init__(self, tol: float = 1e-4, max_iter: int = 100) -> None:
    self.tol = tol
    self.max_iter = max_iter

  def fit(self, x_train: NumNPArrayNxM, y_train: ArrayLike, alpha: float = 0.001) -> None:
    self.K = np.unique(y_train).shape[0]  #Number of classes
    K = self.K
    #Add a column of 1's at the beginning of the data to calculate y intercept
    X = np.c_[np.ones((x_train.shape[0],1)),x_train]
    N = X.shape[0]  #Number of object in training set
    P = X.shape[1]  #Number of parameters + 1 because of the 1's column
    #Reshape y_train so it is a 1 column matrix
    Y = to_categorical(y_train).reshape((-1,1))
    #Set beta values to 0
    B = np.zeros(shape=(K*P,1))
    new_B = np.empty(shape=(K*P,1))
    for j in range(self.max_iter):
      #Calculate the probabilities
      p = np.empty((N,self.K))
      for i in range(K):
        #p[:,i] = np.reshape(np.exp(np.dot(X,B[i::K])),(-1))
        p[:,i] = np.reshape(np.exp(np.dot(X,B[i*P:i*P+P])),(-1))
      p /= np.sum(p,axis=1).reshape((-1,1))
      p = np.reshape(p,(-1,1))
      #Calculate new betas
      y_p = Y-p
      for i in range(K):
        #W = p[i*N:i*N+N]
        W = p[i::K]
        W = W*(1.-W)
        XTWX = np.matmul(X.T,W*X)
        temp = np.matmul(np.linalg.pinv(XTWX),X.T)
        new_B[i*P:i*P+P] = B[i*P:i*P+P] + np.matmul(temp,y_p[i::K])
        #new_B[i::K] = B[i::K] + np.matmul(temp,y_p[i::K])
        #new_B[i::K] = B[i::K] + np.matmul(temp,y_p[i*N:i*N+N])
        #new_B[i*P:i*P+P] = B[i*P:i*P+P] + np.matmul(temp,y_p[i*N:i*N+N])
      diff = B.reshape((P,K)) - new_B.reshape((P,K))
      print(np.all(np.sum(abs(diff)) < self.tol))
      if np.all(np.sum(abs(diff)) < self.tol):
        print(j)
        break
      B = new_B
    self.intercept_ = np.array(B[0::P])
    #Get all betas that are not the intercepts and store it as a 2D array, 1 row per class coefficients
    self.coef_ = np.delete(B,list(range(0,B.shape[0],P)),axis=0).reshape((-1,P-1))

  def predict(self, x_test: NumNPArrayNxM) -> NumNPArray:
    X = np.c_[np.ones((x_test.shape[0],1)),x_test]
    B = np.hstack((self.intercept_,self.coef_)).reshape((-1))
    p = np.reshape(self.__P(X,B),(-1,self.K))
    return np.argmax(p,axis=1)

  def __str__(self) -> str:
    return '{}(max_iter={:03})'.format(self.__class__.__name__,self.max_iter)

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gamma_dataset()
  #x_train,x_test,y_train,y_test = mnist()

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
