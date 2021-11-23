import numpy as np
from scipy.interpolate import RectBivariateSpline

def AffineLucasKanade(It, It1, rect, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3)
    
    p = M[0:2, 0:3].flatten()
    
    # x1, y1, x2, y2 = 0, 0, It.shape[1] -1, It.shape[0] -1
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf")]* 6)
    
    count = 0

    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        
        count += 1

        Itx = It_spline.ev(Y, X)
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        # commonpts = np.where((warpX >= x1) & (warpX <= x2) & 
        #                      (warpY >= y1) & (warpY <= y2), True, False) 
        
        # X, Y = X[commonpts], Y[commonpts]
        # warpX, warpY = warpX[commonpts], warpY[commonpts]
        
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
        It1x_warp = It1_spline.ev(warpY, warpX)
        
        A = np.array([dwx*X.flatten(),
                      dwx*Y.flatten(),
                      dwx,
                      dwy*X.flatten(),
                      dwy*Y.flatten(),
                      dwy]).T
        
        # b = (It[commonpts] - It1x_warp).flatten()
        b = (Itx - It1x_warp).flatten()
        # print(np.shape(A))
    
        
        # print(np.shape(b))
        
        dp = (np.linalg.inv(A.T @ A)) @ (A.T @ b)

        # H = A.T @ A
        # dp = np.linalg.inv(H) @ A.T @ (It[commonpts] - It1x_warp)
        # print(np.shape(dp))
        p+=dp
        
    M = np.vstack((p.reshape(2,3), M[2,0:3]))

    return p
