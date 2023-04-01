import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def process_gamma_dataset():
  #Read file information
  infile = open("gamma04.txt","r")
  x, y  = [], []
  for line in infile:
    y.append(int(line[-2:-1] =='g'))
    x.append(np.fromstring(line[:-2], dtype=float, sep=','))
  infile.close()

  #Make dependent and independent numpy arrays
  x = np.array(x).astype(np.float32)
  y = np.array(y)

  #Split the data
  return train_test_split(x, y, test_size=0.2, random_state=4361)

def calc_predict(y_train):
  #Get all possible values of y_train
  options = np.unique(y_train)

  #Calculate which options gives the highest accuracy
  largest_sum = -1
  best_option = -1
  for option in options:
    curr_sum = np.sum(y_train==option)
    if largest_sum < curr_sum:
      largest_sum = curr_sum
      best_option = option

  return (best_option, largest_sum)

def best_feature_DT(x_train, y_train):
  best_feature = -1
  best_acc = -1.0
  for i_feature in range(x_train.shape[1]):
    sub_x = x_train[:,i_feature]
    i_mean = np.mean(sub_x)
    l_temp = calc_predict(y_train[sub_x<=i_mean])
    r_temp = calc_predict(y_train[sub_x>i_mean])
    curr_acc = (l_temp[1] + r_temp[1])/y_train.shape[0]
    if best_acc < curr_acc:
      best_acc = curr_acc
      best_feature = i_feature
  return best_feature

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
    if isinstance(x_train[0,0], np.floating):
      self.isnumeric = True

      #Split using the mean
      self.break_point = np.mean(sub_x)

      for i in range(2):
        new_depth = None if self.max_depth == None else self.max_depth-1
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
        new_depth = None if self.max_depth == None else self.max_depth-1
        dt = DecisionTreeClassifier(splitter=self.splitter, max_depth=new_depth)

        #Add dt as a branch
        self.branches[option] = dt

        new_idxs = sub_x==option

        #If max_depth is 1 or if only one y value is left, we create the leafs
        new_y = y_train[new_idxs]
        if self.max_depth == 1 or np.unique(new_y).shape[0]==1:
          #Set dt as a leaf and fill in its attributes
          dt.is_leaf = True
          dt.pred_val = calc_predict(new_y)[0]
        else:
          dt.fit(x_train[new_idxs], new_y)

  def fit(self, x_train, y_train, feature_idx=None):
    #If max_depth is less than 1 we fail
    if self.max_depth != None and self.max_depth < 1:
      raise InvalidParameterError("max_depth must be in the range [1,inf)")

    #feature_idx is out of bound we fail
    if feature_idx != None and (feature_idx < 0 or feature_idx > x_train.shape[1]):
      raise InvalidParameterError(f"feature_idx must be in the range [0,{x_train.shape[1]}]")

    #Use feature index given or use splitter option to get one
    if (feature_idx == None):
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
      for dt in self.branches.values():
        match_x = sub_x==dt.att_value
        #Select x_test object that match this branch and assign prediction values
        pred[match_x] = dt.predict(x_test[match_x])
    return pred

def one_feature_DT(x_train, y_train, a):

if __name__ == '__main__':
  ## Process gamma ray dataset
  x_train, x_test, y_train, y_test = process_gamma_dataset()

  ##Part a
  feature_idx = int(input("What feature index you want to try? "))
  model = DecisionTreeClassifier(max_depth=1)
  model.fit(x_train, y_train, a)
  pred = model.predict(x_test)
  print("a: model accuracy = ", accuracy(y_test, pred))

  ##Part b
  print("With best feature split:")
  model = DecisionTreeClassifier(splitter="best", max_depth=1)
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  print("b: model accuracy = ", accuracy(y_test, pred))
  print("b: Best feature = ", model.feature_idx)

