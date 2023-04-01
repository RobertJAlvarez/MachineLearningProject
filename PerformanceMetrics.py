import numpy as np

def precision(y,p):
  return np.sum(y*p)/np.sum(p)
  
def recall(y,p):
  return np.sum(y*p)/np.sum(y)

def f1(y,p):
  pr =precision(y,p)
  r = recall(y,p)
  return 2*pr*r/(pr+r)
  
def accuracy(y,p):
  return np.sum(y==p)/len(y)

def specificity(y,p):
  return np.sum((1-y)*(1-p))/np.sum(y==p)

def confusion_matrix(y,p):
  y_u = np.unique(y)
  m = []
  for i in y_u:
    p_i = p[y==i]
    arr = []
    for j in y_u:
      arr.append(np.sum(p_i==j))
    m.append(arr)
  return m

