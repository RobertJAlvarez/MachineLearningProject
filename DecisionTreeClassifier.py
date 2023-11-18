import numpy as np
from PerformanceMetrics import accuracy
from DataSets import process_gamma_dataset, mnist
from sklearn.tree import DecisionTreeClassifier as DTC
from time import time

def best_feature_DT(x_train, y_train):
  best_feature = -1
  best_acc = -1.0
  for i_feature in range(x_train.shape[1]):
    sub_x = x_train[:,i_feature]
    if isinstance(sub_x[0], (np.integer, np.floating)):
      i_mean = np.mean(sub_x)
      l_temp = calc_predict(y_train[sub_x<=i_mean])
      r_temp = calc_predict(y_train[sub_x>i_mean])
      curr_acc = (l_temp[1] + r_temp[1])/y_train.shape[0]
    else:
      curr_acc = 0.0
      for option in np.unique(sub_x):
        curr_acc += calc_predict(y_train[sub_x==option])[1]
      curr_acc /= y_train.shape[0]

    if best_acc < curr_acc:
      best_acc = curr_acc
      best_feature = i_feature
  return best_feature

def calc_predict(y_train):
  #Get all possible values of y_train

  #Calculate which options gives the highest accuracy
  largest_sum = -1
  best_option = -1
  for option in np.unique(y_train):
    curr_sum = np.sum(y_train==option)
    if largest_sum < curr_sum:
      largest_sum = curr_sum
      best_option = option

  return (best_option, largest_sum)

class InvalidParameterError(Exception):
  def __init__(self, messate="Invalid parameter"):
    print(messate)

class DecisionTreeClassifier:
  def __init__(self, splitter="best", max_depth=None):
    #Decision tree cannot be a leaf unless it have a predicted value assigned
    self.is_leaf = False
    self.splitter = splitter
    self.max_depth = max_depth

  def split_feature_DT(self, x_train, y_train, feature_idx):
    self.branches = dict()
    self.feature_idx = feature_idx
    #Get all rows for feature selected
    sub_x = x_train[:,feature_idx]
    #If the attribute is numeric we use the mean, if it is categorical we use the unique values
    if isinstance(sub_x[0], (np.integer, np.floating)):
      self.isnumeric = True
      #Split using the mean
      self.break_point = np.mean(sub_x)
      for i in range(2):
        new_depth = None if self.max_depth is None else self.max_depth-1
        dt = DecisionTreeClassifier(splitter=self.splitter, max_depth=new_depth)
        #Add dt into the tree and get the indices for the left or right branch
        if i == 0:
          new_idxs = sub_x <= self.break_point
          self.branches["le"] = dt
        else:
          new_idxs = sub_x > self.break_point
          self.branches["g"] = dt
        #If max_depth is 1 or if only one y value is left, we create the leafs
        new_y = y_train[new_idxs]
        if self.max_depth == 1 or np.unique(new_y).shape[0]==1:
          #Set dt as a leaf and fill in its attributes
          dt.is_leaf = True
          dt.pred_val = calc_predict(new_y)[0]
        else:
          dt.fit(x_train[new_idxs], new_y)
    #If the attribute is categorical we use its different possible values
    else:
      self.isnumeric = False
      #Get all possible classification values from sub_x
      options = np.unique(sub_x)
      #Make a branch for each option
      for option in options:
        new_depth = None if self.max_depth is None else self.max_depth-1
        dt = DecisionTreeClassifier(splitter=self.splitter, max_depth=new_depth)
        #Add dt as a branch
        self.branches[option] = dt
        new_idxs = sub_x==option
        new_y = y_train[new_idxs]
        #If max_depth is 1 or if only one y value is left, we create the leafs
        if self.max_depth == 1 or np.unique(new_y).shape[0]==1:
          #Set dt as a leaf and fill in its attributes
          dt.is_leaf = True
          dt.pred_val = calc_predict(new_y)[0]
        else:
          dt.fit(x_train[new_idxs], new_y)

  def fit(self, x_train, y_train, feature_idx=None):
    #If max_depth is less than 1 we fail
    if self.max_depth is not None and self.max_depth < 1:
      raise InvalidParameterError("max_depth must be in the range [1,inf)")

    #feature_idx is out of bound we fail
    if feature_idx is not None and (feature_idx < 0 or feature_idx > x_train.shape[1]):
      raise InvalidParameterError(f"feature_idx must be in the range [0,{x_train.shape[1]}]")

    #Use feature index given or use splitter option to get one
    if (feature_idx is None):
      if self.splitter == "best":
        best_feature = best_feature_DT(x_train, y_train)
        self.split_feature_DT(x_train, y_train, best_feature)
      elif self.splitter == "random":
        self.split_feature_DT(x_train, y_train, np.random.randint(0, x_train.shape[1], 1)[0])
    #Else use the feature column that is given
    else:
      self.split_feature_DT(x_train, y_train, feature_idx)

  def predict(self, x_test):
    #Make prediction array
    pred = np.empty(shape=(x_test.shape[0]))

    if self.is_leaf:
      pred.fill(self.pred_val)
      return pred

    #Create sub_x with the column of interest
    sub_x = x_test[:,self.feature_idx]

    #For this simple case we only have leafs
    if self.isnumeric:
      #Do a recursive prediction for both branches
      l_sub = sub_x<=self.break_point
      pred[l_sub] = self.branches["le"].predict(x_test[l_sub])
      r_sub = sub_x > self.break_point
      pred[r_sub] = self.branches["g"].predict(x_test[r_sub])
    else:
      for option,dt in self.branches.items():
        match_x = sub_x==option
        #Select x_test object that match this branch and assign prediction values
        pred[match_x] = dt.predict(x_test[match_x])
    return pred

if __name__ == '__main__':
  ## Process gamma ray dataset
  #x_train, x_test, y_train, y_test = process_gamma_dataset()
  x_train, x_test, y_train, y_test = mnist()

  ##Part a
  print("My model")
  t0 = time()
  model = DecisionTreeClassifier(splitter="best", max_depth=10)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.3f}".format(time() - t0))
  print("a: model accuracy = ", accuracy(y_test, pred))

  print("SKLEARN model:")
  t0 = time()
  model = DTC(splitter="best", max_depth=10)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("Elapse time = {:.3f}".format(time() - t0))
  print("a: model accuracy = ", accuracy(y_test, pred))

  ##Part b
  #print("With best feature split:")
  #model = DecisionTreeClassifier(splitter="best", max_depth=1)
  #model.fit(x_train, y_train)
  #pred = model.predict(x_test)
  #print("b: model accuracy = ", accuracy(y_test, pred))
  #print("b: Best feature = ", model.feature_idx)

