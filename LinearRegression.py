import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from PerformanceMetrics import accuracy
from tensorflow.keras.utils import to_categorical
from DataSets import mnist

class LinearRegression:
  def __init__(self):
    pass

  def fit(self,x_train,y_train):
    #Add column of ones to also calculate beta_0
    self.X = np.hstack((np.ones(shape=(x_train.shape[0],1)),x_train))
    self.ny_cols = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    self.coef_ = np.empty(shape=(self.ny_cols,x_train.shape[1]))
    self.intercept_ = np.empty(shape=(self.ny_cols))

    y_train = np.reshape(y_train,(-1,self.ny_cols))

    #Calculate beta coefficients
    temp = np.linalg.pinv(np.matmul(self.X.T, self.X)) #(X^TX)^-1
    temp = np.matmul(temp,self.X.T)

    #Calculate linear regression for each column in y (if one then is for regression, if more than one is for classification
    for i,y_col in enumerate(y_train.T[:]):
      betas = np.matmul(temp,y_col)

      #Extract y intercept and update coef_
      self.intercept_[i] = betas[0]
      self.coef_[i] = betas[1:]

    if self.ny_cols == 1:
      self.coef_ = np.reshape(self.coef_,(-1))
      self.intercept_ = np.reshape(self.intercept_,(-1))

  def predict(self,x_test):
    if self.ny_cols == 1:
      pred = np.matmul(x_test,self.coef_) + self.intercept_
    else:
      pred = np.empty(shape=(x_test.shape[0],self.ny_cols))
      for i,temp in enumerate(zip(self.coef_,self.intercept_)):
        coef = temp[0]
        intercept = temp[1]
        pred[:,i] = np.matmul(x_test,coef) + intercept
    return pred

def classification_():
  x_train,x_test,y_train,y_test = mnist()
  model = LinearRegression()
  y_train_oh = to_categorical(y_train, 10)

  #X = np.hstack((np.ones(shape=(x_train.shape[0],1)),x_train))
  #temp = np.linalg.pinv(np.matmul(X.T, X)) #(X^TX)^-1

  model.fit(x_train,y_train_oh)
  pred = np.argmax(model.predict(x_test),axis=1)

  print("model.coef_.shape = ", model.coef_.shape)
  print("pred.shape = ", pred.shape)
  print("Accuracy = ", accuracy(y_test,pred))

def regression_():
  #Read data
  df = pd.read_csv("Pecan.txt", delimiter="\t")
  #Remove first and last column
  X = df.values[:, range(1, len(df.columns)-1)]
  Y = df.values[:, len(df.columns)-1]

  model = LinearRegression()
  model.fit(X,Y)
  print("y intercept: ", model.intercept_)
  print("coefficients: ", model.coef_)

  #Try to predict from made up data
  newData = [[120,5,80], [20,40,15]]
  print("model predictions:", model.predict(newData))

if __name__ == '__main__':
  regression_()
  classification_()

