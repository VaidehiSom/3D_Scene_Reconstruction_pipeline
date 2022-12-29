import numpy as np

def least_squares_estimation(X1, X2):
  """
  input: X1, X2- two N ×3 matrices of matched, calibrated points
  output: E
  """

  N = np.shape(X1)[0]
  # A = np.zeros((N,9))

  # for i in range (0, N):
  #   # A[i] = [X1[:,0] @ X2[:,0].T, X1[:,0] @ X2[:,1].T, X1[:,0] @ X2[:,2].T, X1[:,1] @ X2[:,0].T, X1[:,1] @ X2[:,1].T, X1[:,1] @ X2[:,2].T, X1[:,2] @ X2[:,0].T, X1[:,2] @ X2[:,1].T, X1[:,2] @ X2[:,2].T]

  #   A[i,0:3] = X1[i,:][0] * X2[i,:].T
  #   A[i,3:6] = X1[i,:][1] * X2[i,:].T
  #   A[i,6:9] = X1[i,:][2] * X2[i,:].T
  # [_,_,V] = np.linalg.svd(A)

  # # print(np.shape(U))
  # # print(np.shape(V))
  # # print(np.shape(A))
  # # print(np.shape(a))

  # E = V.T[:,-1]
  # E = E.T
  # E = E.reshape(3,3)
  # # E = E/E[-1,-1]
  # [U,_,V] = np.linalg.svd(E)

  # a = np.diag([1, 1, 0])
  # # a = np.zeros((np.shape(U)[0], np.shape(V)[0]))
  # # a[0][0] = 1
  # # a[1][1] = 1
  # # a[2][2] = 0
  # E = U @ a @ V.T
  # # print(np.shape(E))

  a = np.zeros((N,9))
  for i in range(N):
    p = X1[i,:]
    q = X2[i,:]
    a[i,0:3] = p[0]*q.T
    a[i,3:6] = p[1]*q.T
    a[i,6:9] = p[2]*q.T
  
  [U, S, Vt] = np.linalg.svd(a)
  E_ = np.transpose(Vt)[:,-1]
  E1 = np.zeros((3,3))
  E1[:,0] = E_[0:3]
  E1[:,1] = E_[3:6]
  E1[:,2] = E_[6:9]
  [u, s, v] = np.linalg.svd(E1)
  E = u @ np.diag([1,1,0]) @ v
  
  
  return E
