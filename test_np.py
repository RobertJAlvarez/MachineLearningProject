import numpy as np
from Utils import to_categorical

N = 3
K = 2
P = 3
NK = N*K

def P_(X,B):
  #a = np.empty((N,K))
  #for i in range(K):
  #  a[:,i] = np.reshape(np.exp(np.dot(X,B[i*(P+1):i*(P+1)+P+1])),(-1))
  #a = np.reshape(a.T,(-1,1))
  a = np.empty((NK,1))
  for i in range(K):
    a[i*N:i*N+N] = np.exp(np.dot(X,B[i*(P+1):i*(P+1)+P+1]))
  return a/np.sum(a)

Y = np.array([0,1,0])
Y = to_categorical(Y).reshape((-1,1))
#print("Y.shape = ", Y.shape)
X = np.reshape(np.arange(51,N*(P+1)+51), (N,P+1))
#print("X.shape = ", X.shape)

B = np.zeros(shape=(K*(P+1),1))
p = P_(X,B)
print(p)
#print("p.shape = ", p.shape)

W = np.reshape(np.arange(1,NK*N+1), (NK,N))
#print("W.shape = ", W.shape)

XTWX = np.empty(shape=(P+1,P+1))
#temp = np.empty(shape=(K*(P+1),N))
y_p = Y-p
ans = np.empty(shape=(K*(P+1),1))
for i in range(K):
  W = p[i*N:i*N+N]*(1-p[i*N:i*N+N])
  XTWX = np.matmul(X.T,W*X)
  temp = np.matmul(np.linalg.pinv(XTWX),X.T)
  ans[i*(P+1):i*(P+1)+P+1] = np.matmul(temp,y_p[i*N:i*N+N])

print("new_beta.shape = ", ans.shape)
print(ans)

