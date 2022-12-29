import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 10):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
        (for both sampling and finding inliers)
        params:
            @u: np.array(h,w)
            @v: np.array(h,w)
            @smin: np.array(h,w)
        return value:
            @best_ep: np.array(3,)
            @inliers: np.array(n,) 
        
        u, v and smin are (h,w), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''
    X = np.linspace(-256,255,512)
    xp, yp = np.meshgrid(X, X)

    smin_flatten = smin.flatten()
    orig_indices = np.where(smin_flatten > thresh)[0]
    num = len(orig_indices)

    xp_t = xp.flatten()[smin_flatten > thresh]
    yp_t = yp.flatten()[smin_flatten > thresh]
    Xp = np.vstack((xp_t, yp_t, np.ones(num)))

    uT = u.flatten()[smin_flatten > thresh]
    vT = v.flatten()[smin_flatten > thresh]
    U = np.vstack((uT, vT, np.zeros(num)))   

    cross_p = np.cross(Xp.T, U.T)
    sample_size = 2
    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations):
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above

        cross_p_sampled = cross_p[sample_indices]
        inliers = orig_indices[sample_indices]
        cross_p_test = cross_p[test_indices]
        
        _, _, vh = np.linalg.svd(cross_p_sampled)
        ep = np.transpose(vh)[:,-1]

        dist = np.abs(cross_p_test@ep)
        inliers = np.append(inliers, orig_indices[test_indices[np.where(dist< eps)[0]]])

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep, best_inliers