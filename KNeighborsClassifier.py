import numpy as np
from PerformanceMetrics import accuracy
from DataSets import mnist
from sklearn.neighbors import KNeighborsClassifier as KNN
from NearestNeighbor import NearestNeighbor
from Utils import NumNPArrayNxM, NumNPArray

class KNeighborsClassifier(NearestNeighbor):
  """
  Extension of NearestNeighbor to be use as a classifier. Classification is done
  by chossing the most common target value from the k nearest neighbors.
  """
  # Predict class of test data
  def predict(self, x_test: NumNPArrayNxM) -> NumNPArray:
    """
    Get k nearest neighbors by calling get_closest_k from the super class with
    x_test as parameter. For unweighted distance, use those indeces to the k closes
    elements to get the mode out of the y value of those indeces. For weighted
    distance, use those indeces to get the target values and predict the maximum
    argument of the class with largest number where the number of each class
    is calculates by adding 1/dist(x,x'') to the class where x'' belongs to.

    Args:
        x_test (NumNPArrayNxM): _description_

    Returns:
        NumNPArray: _description_
    """
    # Distance matrix is created, get the k closest elements
    minIdxs, dist = super().get_closest_k(x_test)

    # Build weight array with the votes and choose the one with the highest one to predict
    possibles = np.zeros((dist.shape[0],self.classes_.shape[0]))
    for irow,rowIdx in enumerate(minIdxs):
      for idx in rowIdx:
        if self.weighted:
          possibles[irow,self.y_train[idx]] += 1/dist[irow,idx]
        else:
          possibles[irow,self.y_train[idx]] += 1

    return self.classes_[np.argmax(possibles, axis=1)]

def run_knn_model(knn_model: KNeighborsClassifier | KNN) -> None:
  for weighted in [False,True]:
    for k in range(1,16,2):
      print('k =',k,'weighted =',weighted)
      model = knn_model(k=k, weighted = weighted)
      model.fit(x_train, y_train)
      pred = model.predict(x_test)
      print('Accuracy = {:6.4f}'.format(accuracy(y_test, pred)))

if __name__ == '__main__':
  x_train,x_test,y_train,y_test = mnist()

  print("My model:")
  run_knn_model(KNeighborsClassifier)

  print("SKLEARN model:")
  run_knn_model(KNN)
