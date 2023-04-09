import numpy as np
from Utils import to_categorical

N = 3
K = 2
P = 3
NK = N*K

def P_(X,B):
  a = np.empty((N,K))
  for i in range(K):
    a[:,i] = np.reshape(np.exp(np.dot(X,B[i*(P+1):i*(P+1)+P+1])),(-1))
  a /= np.sum(a,axis=0)
  return np.reshape(a,(-1,1))

#Generate values of X and Y
Y = np.array([0,1,0])
Y = to_categorical(Y).reshape((-1,1))
X = np.reshape(np.arange(51,N*(P+1)+51), (N,P+1))

#Set beta values to 0
B = np.zeros(shape=(K*(P+1),1))
#Get probabilities for each object
p = P_(X,B)
#Calculate y-p
y_p = Y-p
#Calculate new betas
for i in range(K):
  W = p[i*N:i*N+N]*(1-p[i*N:i*N+N])
  XTWX = np.matmul(X.T,W*X)
  temp = np.matmul(np.linalg.pinv(XTWX),X.T)
  B[i*(P+1):i*(P+1)+P+1] -= np.matmul(temp,y_p[i*N:i*N+N])

#Print results
print("new_beta.shape = ", B.shape)
print("new_betas = \n", B)

