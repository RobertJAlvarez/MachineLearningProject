import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNN
from time import time

class KNeighborsRegressor:
  """
  Extension of NearestNeighbor to be use as a regressor. Regression is done by
  getting the k closest point at taking the average of each dimension.
  """
  # Constructor
  def __init__(self,k=1,weighted=False):
    super().__init__(k=k,weighted=weighted)

  # Fit model parameters to training data
  def fit(self,x_train,y_train):
    super().fit(x_train,y_train,regression=True)

  # Predict class of test data
  def predict(self,x_test):
    """Prediction is done with the closest k points by taking the average of each
    dimension for unweighted distance and the average with the weight of
    1/dist(x,x'') for weighted distance.

    Args:
        x_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Distance matrix is created, get the k closest elements
    minIdxs, dist = super().get_closest_k(x_test)

    if self.weighted:
      weights = np.array([1/dist[irow,rowIdx] for irow,rowIdx in enumerate(minIdxs)])
      ans = np.average(self.y_train[minIdxs],axis=1,weights=weights)
    else:
      ans = np.mean(self.y_train[minIdxs],axis=1)
    return ans

if __name__ == '__main__':
  print("Using Pecan.txt for regression:")
  #Read data
  df = pd.read_csv("./datasets/Pecan.txt", delimiter="\t")

  #Remove first and last column
  X = df.values[:, range(1, len(df.columns)-1)]
  Y = df.values[:, len(df.columns)-1]
  newData = np.array([[120,5,80], [20,40,15]])

  for k in range(1,6,2):
    print('k =',k)
    print("My model:",end=' ')
    t0 = time()
    model = KNeighborsRegressor(k=k,weighted=True)
    model.fit(X,Y)
    print("Prediction = ", model.predict(newData))
    print("Elapse time = {:.5f}".format(time() - t0))

    print("SKLEARN model:",end=' ')
    t0 = time()
    model = KNN(n_neighbors=k,weights='distance')
    model.fit(X,Y)
    print("Prediction = ", model.predict(newData))
    print("Elapse time = {:.5f}".format(time() - t0))
