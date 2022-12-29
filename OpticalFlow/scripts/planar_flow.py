import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """
    K_inv = np.linalg.inv(K)
    u = flow_x/K[0,0]/K[2,2]
    v = flow_y/K[1,1]/K[2,2]

    A=[]
    B=[]

    for i in range(up[0],down[0]):
        for j in range(up[1],down[1]):
            X_inv = K_inv @ np.array([j,i,1])
            x = X_inv[0] / X_inv[2]
            y = X_inv[1] / X_inv[2]
 
            A.append([x**2, x*y, x, y, 1, 0, 0, 0])
            B.append([u[i,j]])
            A.append([x*y, y**2, 0, 0, 0, y, x, 1])
            B.append([v[i,j]])
    
    sol = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0]
    return sol.flatten()
    
