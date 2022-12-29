import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """
    flow_x=[]
    flow_y=[]
    x=[]
    y=[]

    for w in range(confidence.shape[1]):
        for h in range(confidence.shape[0]):
            if (confidence[h,w]>threshmin):
                flow_x.append([flow_image[h,w,0]])
                flow_y.append([flow_image[h,w,1]])
                x.append(w)
                y.append(h)
    
    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)
    x = np.array(x)
    y = np.array(y)
    
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, (flow_x*10).astype(int), (flow_y*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    

