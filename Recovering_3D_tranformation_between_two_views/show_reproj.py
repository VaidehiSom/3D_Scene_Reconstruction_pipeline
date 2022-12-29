import numpy as np
import matplotlib.pyplot as plt

def show_reprojections(image1, image2, uncalibrated_1, uncalibrated_2, P1, P2, K, T, R, plot=True):

  """ YOUR CODE HERE
  """
  N = np.shape(uncalibrated_1)[1]
  P1proj = np.zeros((N,3))
  P2proj = np.zeros((N,3))
  # print(np.shape(P1))
  # print(np.shape(uncalibrated_1))
  # print(np.shape(T))
  # print(np.shape())
  # print(np.shape())

  for i in range(N):
    # P1proj[i,:] = np.linalg.inv(K) @ (R @ P1.T[:,i] + T)
    # P2proj[i,:] = np.linalg.inv(K) @ np.linalg.inv(R) @ (P2.T[:,i] - T)
    P1proj[i,:] = K @ (R @ P1.T[:,i] + T)
    P2proj[i,:] =  K @ np.linalg.inv(R) @ (P2.T[:,i] - T)
    # P1proj[i,:] = K @  (R @ P1.T[:,i] + T) @ np.linalg.inv(K)
    # P2proj[i,:] = K @  np.linalg.inv(R) @ (P2.T[:,i] - T) @ np.linalg.inv(K)
    # P1proj[i,:] = R @ P1.T[:,i] + T
    # P2proj[i,:] = np.linalg.inv(R) @ (P2.T[:,i] - T)
    
    # P1proj[i,:] = P1proj[i,:] / P1proj[i,-1]
    # P2proj[i,:] = P2proj[i,:] / P2proj[i,-1]
 
  # P1proj = P1proj[:,:-1]
  # P2proj = P2proj[:,:-1]
  """ END YOUR CODE
  """

  if (plot):
    plt.figure(figsize=(6.4*3, 4.8*3))
    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image1[:, :, ::-1])
    plt.plot(P2proj[:, 0] / P2proj[:, 2],
           P2proj[:, 1] / P2proj[:, 2], 'bs')
    plt.plot(uncalibrated_1[0, :], uncalibrated_1[1, :], 'ro')

    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([0, image1.shape[1]])
    ax.set_ylim([image1.shape[0], 0])
    plt.imshow(image2[:, :, ::-1])
    plt.plot(P1proj[:, 0] / P1proj[:, 2],
           P1proj[:, 1] / P1proj[:, 2], 'bs')
    plt.plot(uncalibrated_2[0, :], uncalibrated_2[1, :], 'ro')
    # return P1proj, P2proj
    
  else:
    return P1proj, P2proj