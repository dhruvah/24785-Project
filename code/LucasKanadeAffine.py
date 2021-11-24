import numpy as np
from scipy.interpolate import RectBivariateSpline


# Gauss-Newton
def LucasKanadeAffine_GN(It, It1, rect, threshold, num_iters):
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
    
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf")] * 6)
    
    count = 0
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    Itx = It_spline.ev(Y, X).flatten()

    while np.linalg.norm(dp)>= threshold and count <= num_iters:

        count += 1
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
        It1x_warp = It1_spline.ev(warpY, warpX).flatten()
        
        G = np.array([dwx * X.flatten(),
                      dwx * Y.flatten(),
                      dwx,
                      dwy * X.flatten(),
                      dwy * Y.flatten(),
                      dwy]).T
        
        b = Itx - It1x_warp
        
        H = G.T @ G
        dp = (np.linalg.inv(H)) @ (G.T @ b)
        p += dp

    return p


# Levenberg-Marquardt
def LucasKanadeAffine_LM(It, It1, rect, threshold, num_iters, delta=0.01):
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
    
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf")] * 6)
    
    count = 0
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    Itx = It_spline.ev(Y, X).flatten()
    prev_error = -1

    while np.linalg.norm(dp)>= threshold and count <= num_iters:

        count += 1
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
        It1x_warp = It1_spline.ev(warpY, warpX).flatten()
        
        b = Itx - It1x_warp
        error = np.linalg.norm(b)

        if prev_error != -1:
            if error < prev_error:
                delta /= 10
            else:
                p -= dp
                It1x_warp = It1_spline.ev(y+p[1], x+p[0]).flatten()
                delta *= 10
        prev_error = error

        G = np.array([dwx * X.flatten(),
                      dwx * Y.flatten(),
                      dwx,
                      dwy * X.flatten(),
                      dwy * Y.flatten(),
                      dwy]).T
        
        LM_term = np.diag(np.sum(G**2, axis=0))
        H_LM = G.T @ G + delta * LM_term
        dp = (np.linalg.inv(H_LM)) @ (G.T @ b)
        p += dp

    return p
