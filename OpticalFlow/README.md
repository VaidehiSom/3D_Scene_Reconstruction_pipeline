# Optical flow

## Output
### Inliers for thresmin = 1, 10, 30
Inliers threshmin = 1
![Inliers_1](results/epipole_1.png)\
Inliers threshmin = 10
![Inliers_10](results/epipole_10.png)\
Inliers threshmin = 30
![Inliers_30](results/epipole_30.png)

### Depth map for thresmin = 1, 10, 30
depth threshmin = 1
![depth_1](results/depth_1.png)\
depth threshmin = 10
![depth_10](results/depth_10.png)\
depth threshmin = 30
![depth_30](results/depth_30.png)

### vector field for thresmin = 1, 10, 30
flow threshmin = 1
![flow_1](results/flow_1.png)\
flow threshmin = 10
![flow_10](results/flow_10.png)\
flow threshmin = 30
![flow_30](results/flow_30.png)

## Running
To run this program:

```
python3 main.py [list of arguments]
```
For example
```
python3 main.py --plot_flow
python3 main.py --depth --threshmin 5
```