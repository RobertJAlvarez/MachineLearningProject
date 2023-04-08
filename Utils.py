import numpy as np

def to_categorical(y):
  y_vals = np.unique(y)
  cat_y = np.empty(shape=(y.shape[0],y_vals.shape[0]))

  for i,y_val in enumerate(y_vals):
    cat_y[:,i] = (y==y_val).astype(int)

  return cat_y

