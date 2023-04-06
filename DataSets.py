import numpy as np
import tensorflow as tf
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

def mnist():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = np.float32(x_train.reshape(x_train.shape[0],-1)/255)
  x_test = np.float32(x_test.reshape(x_test.shape[0],-1)/255)
  return x_train,x_test,y_train,y_test

