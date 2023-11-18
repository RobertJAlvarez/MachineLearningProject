from abc import ABC, abstractmethod
import numpy as np
from Utils import NumNPArrayNxM, NumNPArray, ArrayLike

class NearestNeighbor(ABC):
  """
  Abstract class use by KNeighborsClassifier and KNeighborsRegressor.
  ...
  Atributes
  ---------
  k : int
    Number of nearest neighbors to be find when predicting.
  weighted : boolean
    True for weighted and false for unweighted distance calculations.

   Methods
  ---------
  fit(x_train, y_train, regression=False)
    Save x_train and y_train regardless of regression boolean value. If regression=False
    the unique y values are save in classes_ as another attribute of the class.
  get_closest_k(x_test)
    Calculate and return the k nearest neighbors for each x_test base on the x_train
    values saved when calling fit.
  """
  # Constructor
  def __init__(self, k: int = 1, weighted: bool = False) -> None:
    """
    Save parameters k=1 and weighted=False if no parameter are specified.

    Args:
        k (int, optional): Number of nearest neighbor to be consider for predictions. Defaults to 1.
        weighted (bool, optional): True for weighted and false for unweighted distance calculations. Defaults to False.
    """
    self.k = k
    self.weighted = weighted

  # Fit model parameters to training data
  def fit(self, x_train: NumNPArrayNxM, y_train: ArrayLike, regression: bool = False) -> None:
    """
    Save x_train and y_train regardless of regression boolean value. If regression=False
    the unique y values are save in classes_ as another attribute of the class.

    Args:
        x_train (NumNPArrayNxM): _description_
        y_train (ArrayLike): _description_
        regression (bool, optional): _description_. Defaults to False.
    """
    self.x_train = x_train
    self.y_train = y_train
    if not regression:
      self.classes_ = np.unique(y_train)

  # Predict class of test data
  def get_closest_k(self, x_test: NumNPArrayNxM) -> NumNPArray:
    """Calculate and return the k nearest neighbors for each x_test base on the x_train
    values saved when calling fit.

    Args:
        x_test (NumNPArrayNxM): _description_

    Returns:
        NumNPArray: _description_
    """
    # Improved version using the fact that (a-b)**2 = a**2 - 2ab + b**2
    dist1 = np.sum(x_test**2,axis=1,keepdims=True)
    dist2 = -2.*np.matmul(x_test,self.x_train.T)
    # This part is not really necessary, since it does not depend on x_test
    dist3 = np.sum(self.x_train.T**2,axis=0,keepdims=True)
    dist = dist1+dist2+dist3
 
    # Distance matrix is created, get the k closest elements
    return np.argpartition(dist, kth=self.k, axis=-1)[:,:self.k], dist

  @abstractmethod
  def predict(self, x_test: NumNPArrayNxM) -> NumNPArray:
    pass
