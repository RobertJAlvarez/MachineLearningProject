import numpy as np
import numpy.typing as npt

def precision(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate precision between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Ratio between the True Positives and all the points that are classified as Positives.
  """
  return np.sum(y*p)/np.sum(p)

def recall(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate recall between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Measure of our model correctly identifying True Positives.
  """
  return np.sum(y*p)/np.sum(y)

def f1(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate f1 score between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Harmonic mean of the Precision and Recall.
  """
  pr = precision(y,p)
  r = recall(y,p)
  return 2*pr*r/(pr+r)

def accuracy(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate accuracy between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Fraction of correctly classified samples.
  """
  return np.sum(y==p)/len(y)

def specificity(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate specificity between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Recall of the negative class.
  """
  return np.sum((1-y)*(1-p))/np.sum(y==p)

def confusion_matrix(y: npt.NDArray, p: npt.NDArray) -> list[list[int]]:
  """ Compute confusion matrix to evaluate the accuracy of a classification.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      list[list[int]]: Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
  """
  y_u = np.unique(y)
  m = []
  for i in y_u:
    p_i = p[y==i]
    arr = []
    for j in y_u:
      arr.append(np.sum(p_i==j))
    m.append(arr)
  return m

def MAE(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate Mean Absolute Error between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Average of all output errors is returned.
  """
  return np.sum(np.absolute(y-p))/y.shape[0]

def MSE(y: npt.NDArray, p: npt.NDArray) -> np.float64:
  """ Calculate Mean Square Error between ground truth and estimates.

  Args:
      y (npt.NDArray): Ground truth (correct) target values.
      p (npt.NDArray): Estimated targets as returned by a classifier.

  Returns:
      np.float64: Average of all output square errors is returned.
  """
  return np.sum((y-p)**2)/y.shape[0]
