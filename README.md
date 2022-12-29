# 3D RECONSTRUCTION FROM IMAGES

The entire pipeline consists of different parts -
## Using optical flow to get point correspondences and estimate depths.

### Optical flow is computed first. The smallest singular value of spatiotemporal derivative matrix is calculated and only those pixels which are above a threshold value are considered. the flow vectors are shown below 
<img src="results/flow_10.png" alt="drawing" width="340" height="300"/>

### Epipoles after RANSAC and satisfying planar condition equation by a threshold
<img src="results/epipole_10.png" alt="drawing" width="340" height="300"/>

### Depths are then calculated by assuming pure translational motion
<img src="results/depth_10.png" alt="drawing" width="340" height="300"/>

## Reconstruction of 3d scene from 2 views using 2 view sfm

### We first identify important features using SIFT -
<img src="results/SIFT-points.png" alt="drawing" width="400" height="300"/>

### We then match key points using both least square and RANSAC to prove effectiveness of ransac -
<img src="results/Key-pts-using-lst-sq.png" alt="drawing" width="320" height="300"/>
<img src="results/Key-pts-using-RANSAC.png" alt="drawing" width="320" height="300"/>

### The resulting epipolar lines are as follows 
<img src="results/Epipolar-lines.png" alt="drawing" width="600" height="300"/>

### Finally we reproject the points of one image onto the other
<img src="results/Reprojection.png" alt="drawing" width="600" height="300"/>

## Lastly we recreate the 3D model from multi view sfm

### Input views - 
<img src="results/Input-views.png" alt="drawing" width="300" height="300"/>

### Disparity -
<img src="results/Disparity.png" alt="drawing" width="500" height="300"/>

### Disparity and depth after post processing -
<img src="results/Postproc-Disparity-and-depth.png" alt="drawing" width="500" height="300"/>

### L-R Consistency check mask -
<img src="results/L-R-Consistency-Check-Mask.png" alt="drawing" width="300" height="300"/>

### Reconstructed 3d model from 2 views using ZNCC Kernel -
<img src="results/Reconstructed-3d-model-ZNCC.png" alt="drawing" width="300" height="300"/>

### Entire Reconstructed 3d model
<img src="results/Reconstructed-3d-model.png" alt="drawing" width="300" height="300"/>
<img src="results/Reconstructed-3d-model2.png" alt="drawing" width="300" height="300"/>
