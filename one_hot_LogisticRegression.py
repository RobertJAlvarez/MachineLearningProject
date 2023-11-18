import numpy as np
from PerformanceMetrics import accuracy
from Utils import to_categorical, standardize
from DataSets import process_gamma_dataset, mnist
from sklearn.linear_model import LogisticRegression as LR

class LogisticRegression:
  def sigmoid(self, z):
    return 1/(1+np.exp(-z))

  def __init__(self, penalty=None, tol=1e-4, max_iter=100):
    self.penalty = penalty
    self.tol = tol
    self.max_iter = max_iter

  def fit(self, x_train, y_train, alpha=0.001):
    X = np.c_[np.ones((x_train.shape[0],1)),x_train]  #Add a column of 1's at the beginning of the data to calculate y intercept
    ny_cols = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    all_betas = np.zeros((ny_cols,X.shape[1]))
    y = np.reshape(y_train,(-1,ny_cols))
    self.coef_ = np.empty(shape=(ny_cols,x_train.shape[1]))
    self.intercept_ = np.empty(shape=(ny_cols))

    print("self.coef_.shape = ", self.coef_.shape)
    print("self.intercept_.shape = ", self.intercept_.shape)
    print("all_betas.shape = ", all_betas.shape)

    for i,temp in enumerate(zip(all_betas[:],y.T[:])):
      betas = np.reshape(temp[0],(-1,1))
      print("betas.shape = ", betas.shape)
      y_col = np.reshape(temp[1],(-1,1))
      for _ in range(self.max_iter):
        p = self.sigmoid(np.dot(X,betas))
        W = p*(1-p)
        temp = np.linalg.pinv(np.matmul(X.T, W*X))
        new_betas = betas + np.matmul(np.matmul(temp,X.T),(y_col-p))
        if np.sum(abs(new_betas-betas)) < self.tol:
          break
        betas = new_betas
      print("betas.shape = ", betas.shape)
      self.intercept_[i] = betas[0,0]
      self.coef_[i] = betas[1:,0]

  def predict(self, x_test):
    X = np.c_[np.ones((x_test.shape[0],1)),x_test]
    all_betas = np.hstack((np.reshape(self.intercept_,(-1,1)),self.coef_))
    ny_cols = self.intercept_.shape[0]
    if ny_cols == 1:
      z = np.dot(X, all_betas[0])
      pred = (self.sigmoid(z)>0.5).astype(int)
    else:
      pred = np.empty(shape=(ny_cols,x_test.shape[0]))
      for i,betas in enumerate(all_betas[:]):
        z = np.dot(X, betas)
        pred[i] = self.sigmoid(z)
      pred = np.argmax(pred,axis=0)
    return pred

  def __str__(self):
    return '{}(max_iter={:03}, penalty={})'.format(self.__class__.__name__,self.max_iter,self.penalty)

def classification_():
  print("\nCLASSIFICATION:")
  x_train,x_test,y_train,y_test = mnist()
  y_train_oh = to_categorical(y_train)

  x_train = x_train[:100]
  y_train = y_train[:100]
  y_train_oh = y_train_oh[:100]
  x_test = x_test[:100]
  y_test = y_test[:100]

  #My model
  #print("My model")
  #model = LogisticRegression()
  #print("model = ", model)
  #model.fit(x_train, y_train_oh)
  #pred = model.predict(x_test)
  #print("accuracy: ", accuracy(y_test, pred))

  #print("model.intercept_.shape = ", model.intercept_.shape)
  #print("model.coef_.shape = ",model.coef_.shape)

  #Run sklearn simple model
  print("\nSKLEARN model")
  model = LR(max_iter=100, penalty=None, tol=1e-4)
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("accuracy: ", accuracy(y_test, pred))

  print("model.intercept_.shape = ", model.intercept_.shape)
  print("model.coef_.shape = ",model.coef_.shape)

def Binary_classification():
  print("\nBinary classification:")
  x_train,x_test,y_train,y_test = process_gamma_dataset()
  standardize(x_train)
  standardize(x_test)

  #My model
  print("My model")
  model = LogisticRegression(max_iter=10, penalty=None, tol=1e-4)
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("accuracy: ", accuracy(y_test, pred))

  print("model.intercept_.shape = ", model.intercept_.shape)
  print("model.coef_.shape = ",model.coef_.shape)

  #Run sklearn simple model
  print("\nSKLEARN model")
  model = LR(max_iter=10, penalty=None, tol=1e-4)
  print("model = ", model)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("accuracy: ", accuracy(y_test, pred))

  print("model.intercept_.shape = ", model.intercept_.shape)
  print("model.coef_.shape = ",model.coef_.shape)

if __name__ == '__main__':
  #regression_()
  classification_()

