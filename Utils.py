import numpy as np

def to_categorical(y):
  y_vals = np.unique(y)
  cat_y = np.empty(shape=(y.shape[0],y_vals.shape[0]))
  for i,y_val in enumerate(y_vals):
    cat_y[:,i] = (y==y_val).astype(int)
  return cat_y

def standardize(x):
  for i in range(x.shape[1]):
    x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])

class LabelEncoder:
  # Constructor
  def __init__(self):
    self.classes_ = np.empty((0))

  def fit(self, data):
    self.classes_ = np.unique(data)

  def fit_transform(self, data):
    #Get the classes if they haven't being set
    if self.classes_.shape[0] == 0:
      self.fit(data)
    return self.transform(data)

  def transform(self, data):
    if isinstance(data, list):
      data = np.array(data, self.classes_.dtype)
    temp = np.empty(data.shape,np.int32)
    for i,c in enumerate(self.classes_):
      temp[c==data] = i
    return temp

  def inverse_transform(self, data):
    if isinstance(data, list):
      shape = len(data)
    else:
      shape = data.shape
    temp = np.empty(shape,self.classes_.dtype)
    for i,c in enumerate(self.classes_):
      i = np.int32(i)
      temp[i==data] = c
    return temp

def relu(x):
  return np.maximum(np.array([[0]]),x)

def sigmoid(x):
  return 1./(1.+exp(-x))

