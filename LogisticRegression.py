import numpy as np
from PerformanceMetrics import accuracy
from DataSets import process_gamma_dataset
from sklearn.linear_model import LogisticRegression as LR

def standardize(x):
  for i in range(x.shape[1]):
    x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])

class LogisticRegression:
  def sigmoid(self, z):
    return 1/(1+np.exp(-z))

  def __init__(self, penalty=None, max_iter=100):
    self.penalty = penalty
    self.max_iter = max_iter

  #TODO: Upgrade this so it can handle one-hot representations
  def fit(self, x_train, y_train, alpha=0.001):
    X = np.c_[np.ones((x_train.shape[0],1)),x_train]  #Add a column of 1's at the beginning of the data to calculate y intercept
    betas = np.zeros((X.shape[1],1))  #We have as many betas as we have attributes
    y = np.reshape(y_train,(-1,1))      #Reshape y_train so it is a 1 column matrix
    for _ in range(self.max_iter):
      p = self.sigmoid(np.dot(X,betas))
      W = p*(1-p)
      temp2 = W*X
      temp = np.linalg.inv(np.matmul(X.T, W*X))
      betas = betas + np.matmul(np.matmul(temp,X.T),(y-p))
      #betas = betas - alpha*np.dot(X.T, self.sigmoid(np.dot(X,betas)) - y)
    self.betas = betas

  #TODO: Upgrade this so it can handle one-hot representations
  def predict(self, x_test):
    X = np.c_[np.ones((x_test.shape[0],1)),x_test]
    z = np.dot(X, self.betas)
    pred = np.empty(x_test.shape[0])
    for i,val in enumerate(self.sigmoid(z)):
      pred[i] = 1 if val>0.5 else 0
    return pred

  def __str__(self):
    return '{}(max_iter={:03}, penalty={})'.format(self.__class__.__name__,self.max_iter,self.penalty)

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gamma_dataset()
  standardize(x_train)
  standardize(x_test)

  #My model
  print("My model")
  model = LogisticRegression()
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("accuracy: ", accuracy(y_test, pred))

  #Run sklearn simple model
  print("\nSKLEARN model")
  model = LR(max_iter=100, penalty=None)
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("accuracy: ", accuracy(y_test, pred))

