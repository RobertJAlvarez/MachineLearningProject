import numpy as np
from NPTypes import IntNPArray, FloatNPArrayNxM, ArrayLike, FloatNPArray

def to_categorical(y: IntNPArray) -> IntNPArray:
  y_vals = np.unique(y)
  cat_y = np.empty(shape=(y.shape[0],y_vals.shape[0]))
  for i,y_val in enumerate(y_vals):
    cat_y[:,i] = (y==y_val).astype(int)
  return cat_y

def standardize(x: FloatNPArrayNxM) -> FloatNPArrayNxM:
  for i in range(x.shape[1]):
    x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])

class LabelEncoder:
  """Class use to encode y values from 0 to n_classes-1, where n_classes is the
  number of unique y values.

  Attributes
  ----------
  classes_ : numpy array of shape (n_classes,)
    classes_ is a n_classes entries numpy array where each entry have a unique
    y value

   Methods
  ---------
  fit(data)
    Find the k unique values on the data and save them in classes_ attribute.
  transform(data)
    Transform change the targest values to the respective encode number.
  inverse_transform(data)
    Reverse the transformer by mapping the numbers from 0 to n_classes to its
    original value.
  """
  # Constructor
  def __init__(self) -> None:
    self.classes_ = np.empty((0))

  def fit(self, data: ArrayLike) -> None:
    """ Find the k unique values on the data and save them in classes_ attribute.

    Args:
        data (ArrayLike): _description_
    """
    self.classes_ = np.unique(data)

  def fit_transform(self, data: ArrayLike) -> IntNPArray:
    # Get the classes if they haven't being set
    if self.classes_.shape[0] == 0:
      self.fit(data)
    return self.transform(data)

  def transform(self, data: ArrayLike) -> IntNPArray:
    """ It use classes_ to encode each data value to the respective index in the array.

    Args:
        data (ArrayLike): _description_

    Returns:
        IntNPArray: _description_
    """
    if isinstance(data, list):
      data = np.array(data, self.classes_.dtype)
    temp = np.empty(data.shape,np.int32)
    for i,c in enumerate(self.classes_):
      temp[c==data] = i
    return temp

  def inverse_transform(self, data: IntNPArray) -> ArrayLike:
    """ Replace each data values to what is in classes_ by using each data value as an index to classes_.

    Args:
        data (IntNPArray): _description_

    Returns:
        ArrayLike: _description_
    """
    if isinstance(data, list):
      shape = len(data)
    else:
      shape = data.shape
    temp = np.empty(shape,self.classes_.dtype)
    for i,c in enumerate(self.classes_):
      i = np.int32(i)
      temp[i==data] = c
    return temp

def relu(x: FloatNPArray) -> FloatNPArray:
  """ Return the maximum number between 0 and x for each value of x.

  Args:
      x (FloatNPArray): _description_

  Returns:
      FloatNPArray: Maximum number between 0 and x for each value of x.
  """
  return np.maximum(np.array([[0.]]),x)

def sigmoid(x: FloatNPArray) -> FloatNPArray:
  """ Returns 1/(1+exp(-z)) for each value of z.

  Args:
      x (FloatNPArray): _description_

  Returns:
      FloatNPArray: 1/(1+exp(-z)) for each value of z.
  """
  return 1./(1.+np.exp(-x))
