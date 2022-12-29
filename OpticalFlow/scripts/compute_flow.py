import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    flow = np.zeros((2))
    A=[]
    B=[]

    for i in range(size):
      for j in range(size):
        corners_x = (x-2+i < 0) or (x-2 +i >= Ix.shape[1])
        corners_y = (y-2+j < 0) or (y-2 +j >= Ix.shape[0])
        if (corners_y or corners_x):
          continue
        else:
          ax = Ix[y-2+j,x-2+i]
          ay = Iy[y-2+j, x-2+i]
          at = -It[y-2+j,x-2+i]
          A.append([ax, ay])
          B.append(at)

    A = np.array(A)
    B = np.array(B)

    flow[0], flow[1]= np.linalg.lstsq(A, B, rcond=None)[0]
    
    _, S, _ = np.linalg.svd(A)
    conf = np.min(S)
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

