import numpy as np
from PerformanceMetrics import accuracy
from DataSets import mnist
from sklearn.naive_bayes import GaussianNB as GNB
from time import time
from math import pi

class GaussianNB:
  # Constructor
  def __init__(self, var_smoothing=1e-09):
    self.epsilon_ = var_smoothing

  def fit(self, x_train, y_train): # Fit model parameters to training data
    self.classes_ = np.sort(np.unique(y_train))
    self.var_ = np.empty((self.classes_.shape[0],x_train.shape[1]))
    self.theta_ = np.empty((self.classes_.shape[0],x_train.shape[1]))
    pc = np.empty(self.classes_.shape[0])

    for i,y in enumerate(self.classes_):
      # Identify which sample match this class
      ind = y_train==y
      sub_x = x_train[ind]
      self.theta_[i] = np.mean(sub_x,axis=0)
      self.var_[i] = np.std(sub_x,axis=0) + self.epsilon_
      pc[i] = np.mean(ind)
    self.class_prior_ = np.log(pc)

  def predict(self, x_test): # Predict class of test data
    np.seterr(divide = 'ignore')
    probs = np.empty((x_test.shape[0],self.classes_.shape[0]))
    t = 1.0/np.sqrt(2.0*pi)
    for i in range(self.classes_.shape[0]):
      p = (t/self.var_[i])*np.exp(-0.5*((x_test-self.theta_[i])/self.var_[i])**2)
      probs[:,i] = np.sum(np.log(p),axis=1)
    return self.classes_[np.argmax(probs + self.class_prior_,axis=1)]

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = mnist()

  print("My model:")
  t0 = time()
  model = GaussianNB(var_smoothing=1e-6)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

  print("\nSKLEARN model:")
  t0 = time()
  model = GNB(var_smoothing=1e-6)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

