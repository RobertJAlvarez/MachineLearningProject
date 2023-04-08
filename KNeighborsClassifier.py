import numpy as np
from PerformanceMetrics import accuracy
from DataSets import mnist
from sklearn.neighbors import KNeighborsClassifier as KNN

class KNeighborsClassifier:
  # Constructor
  def __init__(self,k=1,weighted = False):
    self.k = k
    self.weighted = weighted

  # Fit model parameters to training data
  def fit(self,x_train,y_train):
    self.x_train = x_train
    self.y_train = y_train
    self.y_unique = np.unique(y_train)

  # Predict class of test data
  def predict(self,x_test):
    # Improved version using the fact that (a-b)**2 = a**2 - 2ab + b**2
    dist1 = np.sum(x_test**2,axis=1,keepdims=True)
    dist2 = -2*np.matmul(x_test,self.x_train.T)
    dist3 = np.sum(self.x_train.T**2,axis=0,keepdims=True) # This part is not really necessary, since it does not depend on x_test
    dist = dist1+dist2+dist3

    # Distance matrix is created, get the k closest elements
    minIdxs = np.argpartition(dist, kth=self.k, axis=-1)[:,:self.k]

    # Build weight array with the votes and choose the one with the highest one to predict
    possibles = np.zeros((dist.shape[0],self.y_unique.shape[0]))
    for irow,rowIdx in enumerate(minIdxs):
      for idx in rowIdx:
        if self.weighted:
          possibles[irow,self.y_train[idx]] += 1/dist[irow,idx]
        else:
          possibles[irow,self.y_train[idx]] += 1

    return self.y_unique[np.argmax(possibles, axis=1)]

def run_knn_model(knn_model):
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

