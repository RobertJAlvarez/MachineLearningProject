import numpy as np
import pandas as pd

class LinearRegression:
  def __init__(self):
    pass

  #TODO: Upgrade this so it can handle one-hot representations
  def fit(self,x_train,y_train):
    #Add column of ones to also calculate beta_0
    self.X = np.hstack((np.ones(shape=(x_train.shape[0],1)),x_train))
    self.y = y_train

    #Calculate beta coefficients
    temp = np.linalg.inv(np.matmul(self.X.T, self.X)) #(X^TX)^-1
    temp = np.matmul(temp,self.X.T)
    self.coef_ = np.matmul(temp,self.y)

    #Calculate y_hat
    self.y_hat = np.matmul(self.X,self.coef_)

    #Extract y intercept and update coef_
    self.intercept_ = self.coef_[0]
    self.coef_ = self.coef_[1:]

  #TODO: Upgrade this so it can handle one-hot representations
  def predict(self,x_test):
    return np.matmul(x_test,self.coef_) + self.intercept_

if __name__ == '__main__':
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

