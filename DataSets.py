import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Binary classification
def process_gamma_dataset():
  #Read file information
  infile = open("./datasets/gamma04.txt","r")
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

#Regression
def process_gpu_running_time():
  df = pd.read_csv('./datasets/gpu_running_time.csv')
  data = df.to_numpy()
  X = data[:,:15]
  y = np.mean(data[:,15:],axis=1)
  return train_test_split(X, y, test_size=0.3, random_state=4361)

#Binary classification
def process_doh():
  # Drop rows that have a NaN in them
  df = pd.read_csv("doh_dataset.csv", compression='gzip').dropna()
  # Remove features we don't want
  # In this case, we remove non-float data as that will mess up the student's code
  df.pop('TimeStamp')
  df.pop('SourceIP')
  df.pop('DestinationIP')
  # Extract the labels from the dataframe and encode them to integers
  df_labels = df.pop('Label')
  label_encoder = LabelEncoder()
  df_labels = label_encoder.fit_transform(df_labels)

  # Prepare arrays to split into training and testing sets
  x_features = df.values
  y_labels = np.array(df_labels).T

  # Split into training (70%) and testing (30%)
  return train_test_split(x_features, y_labels, train_size=0.7, random_state=1738, shuffle=True)

#Multi-classification 
def mnist():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = np.float32(x_train.reshape(x_train.shape[0],-1)/255)
  x_test = np.float32(x_test.reshape(x_test.shape[0],-1)/255)
  return x_train,x_test,y_train,y_test

#Regression
def process_particles():
  PX = np.load('particles_X.npy')
  Py = np.load('particles_y.npy')
  return train_test_split(PX, Py, test_size=0.2,  random_state=4361)

