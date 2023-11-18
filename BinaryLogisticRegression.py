import numpy as np
from PerformanceMetrics import accuracy
from DataSets import process_gamma_dataset
from sklearn.linear_model import LogisticRegression as LR
from Utils import standardize, sigmoid
from time import time
from NPTypes import NumNPArrayNxM, ArrayLike, NumNPArray

class LogisticRegression:
  """
  Moduel use for binary classification task. Optimization use is Newton method.
  ...
  Atributes
  ---------
  self.intercept_ : shape=(1, )
    Real value numbers that represent the bias of each class.
  self.coef_ : shape=(n_attributes, )
    Real value numbers that represent the weights of each class.

   Methods
  ---------
  fit(x_train, y_train)
    Calcualte intercept and beta coefficients using Newtons method.
  predict(x_test)
    Calculate predict values as sigmoid(XB)>0.5.
  """
  def __init__(self, tol: float = 1e-4, max_iter: int = 100) -> None:
    self.tol = tol
    self.max_iter = max_iter

  def fit(self, x_train: NumNPArrayNxM, y_train: ArrayLike, alpha: float = 0.001) -> None:
    """
    Use Newton method for optimization of beta values. Formula to update beta is
    beta_(i+1) = beta_i + (X^T WX)^-1 X^T (y-p)

    Args:
        x_train (NumNPArrayNxM): _description_
        y_train (ArrayLike): _description_
        alpha (float, optional): _description_. Defaults to 0.001.
    """
    X = np.c_[np.ones((x_train.shape[0],1)),x_train]  #Add a column of 1's at the beginning of the data to calculate y intercept
    betas = np.zeros((X.shape[1],1))  #We have as many betas as we have attributes
    y = np.reshape(y_train,(-1,1))    #Reshape y_train so it is a 1 column matrix
    for _ in range(self.max_iter):
      p = sigmoid(np.dot(X,betas))
      W = p*(1-p)
      XTWX = np.matmul(X.T, W*X)
      temp = np.matmul(np.linalg.inv(XTWX),X.T)
      new_betas = betas + np.matmul(temp,(y-p))
      print(np.sum(abs(new_betas-betas)) < self.tol)
      if np.sum(abs(new_betas-betas)) < self.tol:
        break
      betas = new_betas
    self.intercept_ = betas[0,0]
    self.coef_ = betas[1:,0]

  def predict(self, x_test: NumNPArrayNxM) -> NumNPArray:
    """
    Predict y values as sigmoid(XB)>0.5.

    Args:
        x_test (NumNPArrayNxM): _description_

    Returns:
        NumNPArray: _description_
    """
    X = np.c_[np.ones((x_test.shape[0],1)),x_test]
    betas = np.hstack((np.reshape(self.intercept_,(-1)),self.coef_))
    z = np.dot(X,betas)
    return (sigmoid(z)>0.5).astype(int)

  def __str__(self) -> str:
    return '{}(max_iter={:03}, penalty={})'.format(self.__class__.__name__,self.max_iter,self.penalty)

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = process_gamma_dataset()
  standardize(x_train)
  standardize(x_test)

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
