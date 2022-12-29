from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    # highest_count = 0

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])
        # inliers_ = sample_indices
        # inliers_ = np.array((inliers_))
        inliers = []
        # np.concatenate((inliers, X2[sample_indices]), axis=1)
        # count = 0
        # print("before loop")
        for j in test_indices:
            # d_x2 = (X2[j,:].T @ E @ X1[j,:])**2 / np.linalg.norm(np.cross([0,0,1], E @ X1[j,:]))
            # d_x1 = (X1[j,:].T @ E.T @ X2[j,:])**2 / np.linalg.norm(np.cross([0,0,1], E.T @ X2[j,:].T))

            d_x2 = (X2[j].T @ E @ X1[j]) / np.linalg.norm(np.cross([0,0,1], E @ X1[j]))
            d_x1 = (X1[j].T @ E.T @ X2[j]) / np.linalg.norm(np.cross([0,0,1], E.T @ X2[j]))
            residual = d_x1**2 + d_x2**2
            if(residual < eps):
                # count = count+1
                # np.hstack([inliers, test_indices])
                # print("before concatenate")
                inliers.append(j)
                # print("after concatenate")
                # inliers = np.concatenate((inliers, j))

        # if(count > highest_count):
        #     highest_count = count
            # best_E_ = E

        inliers = np.array(inliers)
        inliers = np.concatenate((sample_indices, inliers))
        """ END YOUR CODE
        """
        # print("before final")
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers