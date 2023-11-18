import numpy as np
from PerformanceMetrics import accuracy, MSE
from Utils import to_categorical
from DataSets import mnist, process_gpu_running_time
from sklearn.linear_model import LinearRegression as LR
from time import time

class LinearRegression:
  """
  Module use for regression and classification. If used for classification,
  a one-hot representation neeeds to be pass.
  ...
  Atributes
  ---------
  self.intercept_ : shape=(n_classes, )
    Real value numbers that represent the bias of each class.
  self.coef_ : shape=(n_classes, n_attributes, )
    Real value numbers that represent the weights of each class.

   Methods
  ---------
  fit(x_train, y_train)
    Calcualte beta values as (X^T X)^-1 X^T Y and save them in self.intercept_
    and self.coef_.
  predict(x_test)
    Use the self.intercept_ and self.coef_ to calculate the predicted values as
    X(X^T X)^-1 X^T Y.
  """
  def __init__(self) -> None:
    pass

  #(X.TX)^-1 X.T y
  def fit(self,x_train,y_train):
    #Add column of ones to also calculate beta_0
    X = np.hstack((np.ones(shape=(x_train.shape[0],1)),x_train))
    ny_cols = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    y = np.reshape(y_train,(-1,ny_cols))

    #Calculate beta coefficients
    temp = np.linalg.pinv(np.matmul(X.T, X)) #(X.TX)^-1
    temp = np.matmul(temp,X.T)

    #Calculate linear regression for each column in y
    all_betas = np.matmul(temp,y)
    self.intercept_ = all_betas[0]
    self.coef_ = all_betas[1:].T

    if ny_cols == 1:
      self.coef_ = np.reshape(self.coef_,(-1))
      self.intercept_ = np.reshape(self.intercept_,(-1))

  def predict(self,x_test):
    ny_cols = self.intercept_.shape[0]
    if ny_cols == 1:
      pred = np.matmul(x_test,self.coef_) + self.intercept_
    else:
      pred = np.matmul(x_test,self.coef_.T) + self.intercept_
    return pred

def classification_():
  print("\nUsing mnist for classification:")

  x_train,x_test,y_train,y_test = mnist()
  y_train_oh = to_categorical(y_train)

  print("My model:")
  t0 = time()
  model = LinearRegression()
  model.fit(x_train,y_train_oh)
  pred = np.argmax(model.predict(x_test),axis=1)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

  print("\nSKLEARN model:")
  t0 = time()
  model = LR()
  model.fit(x_train,y_train_oh)
  pred = np.argmax(model.predict(x_test),axis=1)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

def regression_():
  x_train,x_test,y_train,y_test = process_gpu_running_time()

  print("My model:")
  t0 = time()
  model = LinearRegression()
  model.fit(x_train,y_train)
  print("MSE = ", MSE(y_test, model.predict(x_test)))
  print("Elapse time = {:.5f}".format(time() - t0))

  print("\nSKLEARN model:")
  t0 = time()
  model = LR()
  model.fit(x_train,y_train)
  #Try to predict from made up data
  print("MSE = ", MSE(y_test, model.predict(x_test)))
  print("Elapse time = {:.5f}".format(time() - t0))

if __name__ == '__main__':
  regression_()
  classification_()

