from weakref import ref
import numpy as np
import cv2
from scipy.fft import dst
from tqdm import tqdm


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    pnts = np.array(( (0, 0, 1), (width, 0, 1), (0, height, 1), (width, height, 1),), dtype=np.float32).reshape(2, 2, 3)
    pnts = pnts.reshape(4, 3).T

    T = Rt[: 3, -1].reshape((-1, 1))
    R = Rt[: 3, : 3]
    pnts_cam = depth * ((np.linalg.inv(K))  @ pnts) / ((np.linalg.inv(K))  @ pnts)[-1, :]
    pnts_world = (np.linalg.inv(R) @ (pnts_cam - T)).T
    pnts = pnts_world.reshape(2, 2, 3)

    return pnts

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """

    hgt, width = points.shape[0], points.shape[1]
    points = points.reshape(hgt * width, 3)
    a = np.ones((hgt * width, 1))
    points = np.hstack((points, a))
    points_new = K @ Rt @ (points.T)
    points = points_new.T
    points = points / (points[:, -1].reshape((-1 ,1)))
    points = points[:, : -1]
    points = points.reshape(hgt, width, 2)
    # points = points[:-1].T.reshape((points.shape[0], points.shape[1], 2))

    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    srcPnts = backproject_fn(K_ref, width, height, depth, Rt_ref)
    dstPnts = project_fn(K_neighbor, Rt_neighbor, srcPnts)
    dstPnts = dstPnts.reshape((dstPnts.shape[0]*dstPnts.shape[1], 2))

    pnts = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    H, _ = cv2.findHomography(pnts, dstPnts, cv2.RANSAC)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(H), dsize = (width, height))

    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    src_ = np.mean(src, axis = 2)
    src_ = src_[:, :, np.newaxis, :]
    dst_ = np.mean(dst, axis = 2)
    dst_ = dst_[:, :, np.newaxis, :]
    zncc = np.sum((src - src_) * (dst - dst_), axis = 2)/(np.std(src, axis = 2) * np.std(dst, axis = 2) + EPS)
    zncc = np.sum(zncc, axis = 2)

    return zncc


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    u, v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    xcam = (u - K[0, -1]) * dep_map/K[0, 0]
    ycam = (v - K[1, -1]) * dep_map/K[1, 1]
    zcam = dep_map
    xyz_cam = np.dstack((xcam, ycam, zcam))
    return xyz_cam

