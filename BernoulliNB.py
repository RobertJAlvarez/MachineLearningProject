import numpy as np
from PerformanceMetrics import accuracy
from DataSets import mnist
from sklearn.naive_bayes import BernoulliNB as BNB
from time import time

import numpy.typing as npt
from Utils import NumNPArrayNxM

class BernoulliNB:
  """
  Module use to predict y values from binarize X data to calculate conditional
  probabilities as P(y|X) = P(X=1|y)X + (1-P(X=1|y))(1-X).
  ...
  Atributes
  ---------
  self.binarize : float
    Value use to binarize data. If None is given we assume data is already
    binarized.
  self.alpha : float
    Value use as smoother in the conditional probabilities.
  self.classes_ : shape=(n_classes, )
    Numpy array with all classes sorted in ascending order.
  self.pac1 : shape = (n_classes, n_attributes, )
    2D numpy array with the conditional probabilities of such aattribute in
    class being equal to 1.
  self.pac0 : shape = (n_classes, n_attributes, )
    Same as self.pac1 but for value 0.
  self.class_prior_ : shape = (n_classes, )
    Log probability for getting a class.

   Methods
  ---------
  fit(x_train, y_train)
    Calcualte the mean occurance of each class and the probability of each
    attribute beeing 1 or zero given the class.
  predict(x_test)
    Using probability of each class and the the prbability of 1 or 0 for each
    attribute in each class we predict using the following formula:
    P(X|y) = P(X=1|y)X + (1-P(X=1|y))(1-X).
  """
  # Constructor
  def __init__(self, alpha: float = 1.0, force_alpha: bool = False, binarize: float | None = None) -> None:
    self.binarize = binarize
    if alpha is not None:
      self.alpha = 1e-10 if force_alpha and alpha<1e-10 else alpha
    else:
      self.alpha = -1.0

  def fit(self, x_train: NumNPArrayNxM, y_train: npt.ArrayLike) -> None: # Fit model parameters to training data
    if self.binarize is not None:
      x_train = (x_train>self.binarize).astype(int)

    self.classes_ = np.sort(np.unique(y_train))
    pc = np.empty(self.classes_.shape[0])
    self.pac1 = np.empty((self.classes_.shape[0],x_train.shape[1]))

    for i,y in enumerate(self.classes_):
      # Identify which sample match this class
      ind = y_train==y
      # Calculate probability to get 1
      self.pac1[i] = np.mean(x_train[ind],axis=0)
      # Calculate average of how many samples match this class
      pc[i] = np.mean(ind)

    #Add smoothing and re-normalize probabilities
    if self.alpha > 0.0:
      pc += self.alpha # Add smoothing
      self.pac1 = self.pac1*(1-2*self.alpha) + self.alpha
    self.pac0 = 1-self.pac1
    self.class_log_prior_ = np.log(pc)

  def predict(self, x_test): # Predict class of test data
    if self.binarize is not None:
      x_test = (x_test>self.binarize).astype(int)

    x_test_0 = 1.0-x_test
    probs = np.empty((x_test.shape[0],self.classes_.shape[0]))
    for i in range(self.classes_.shape[0]):
      p = self.pac1[i:i+1]*x_test + self.pac0[i:i+1]*x_test_0
      probs[:,i] = np.sum(np.log(p),axis=1)
    return self.classes_[np.argmax(probs + self.class_log_prior_,axis=1)]

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = mnist()
  mean = np.mean(x_train)

  print("My model:")
  t0 = time()
  model = BernoulliNB(alpha=1e-6,binarize=mean)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))

  print("\nSKLEARN model:")
  t0 = time()
  model = BNB(alpha=1e-6,binarize=mean)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.5f}".format(time() - t0))
  print("accuracy: {:.5f}".format(accuracy(y_test, pred)))
