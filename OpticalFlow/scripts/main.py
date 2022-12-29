import argparse
from compute_grad import compute_Ix, compute_Iy, compute_It
from planar_flow import compute_planar_params
from compute_flow import flow_lk
from vis_flow import plot_flow
from depth import depth
import numpy as np
import os
import pdb
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from epipole import epipole

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epipole", action="store_true")
    parser.add_argument("--plot_flow", action="store_true")
    parser.add_argument("--planar_flow", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--threshmin", type=int, default=10)

    args = parser.parse_args()

    # load images
    data_folder = "data"
    images = [cv2.imread(os.path.join(data_folder, "insight{}.png".format(i)), cv2.IMREAD_GRAYSCALE).astype(float)
                                    for i in range(20, 27)]
    images = np.stack(images, axis=-1)

    Ix = compute_Ix(images)
    Iy = compute_Iy(images)
    It = compute_It(images)

    # only take the image in the middle
    valid_idx = 3

    flow, confidence = flow_lk(Ix[..., valid_idx], Iy[..., valid_idx], It[..., valid_idx])

    K = np.array([[1118, 0, 357],
                    [0, 1121, 268],
                    [0, 0, 1]])

    if args.plot_flow:
        plt.figure()
        plot_flow(images[..., valid_idx], flow, confidence, threshmin=args.threshmin)
        plt.savefig(f"flow_{args.threshmin}.png")
        plt.show()

    if args.epipole or args.depth:
        plt.figure()
        block_mask = np.array(confidence)

        ep, inliers = epipole(flow[:,:,0],flow[:,:,1],block_mask,args.threshmin,num_iterations = 1000)

        ep = ep/ep[2]

        plot_flow(images[..., valid_idx], flow, block_mask, threshmin=args.threshmin)

        plt.scatter(ep[0]+512//2, ep[1]+512//2, c='r')

        x = np.array([i for i in range(512)])
        xv, yv = np.meshgrid(x, x)
        xp = np.stack([xv.flatten(),yv.flatten(),np.ones((512,512)).flatten()]).T

        plt.scatter(xp[inliers,:][:,0], xp[inliers,:][:,1], c='b', s=0.1)
        plt.savefig(f"epipole_{args.threshmin}.png")
        plt.show()

    if args.depth:
        depth_map = depth(flow, confidence, ep, K, thres=args.threshmin)
        sns.heatmap(depth_map, square=True, cmap='mako')
        plt.savefig(f"depth_{args.threshmin}.png")
        plt.show()

    if args.planar_flow:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        up = [312, 0]
        down = [512, 200]
        params = compute_planar_params(flow[..., 0], flow[..., 1], K,
                                                up=up, down=down)
        print("8 Arguments are: ", params)










