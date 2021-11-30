import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

def obj_fun(p, X, Y, It_spline, It1_spline):
    warpX = p[0] * X + p[1] * Y + p[2]
    warpY = p[3] * X + p[4] * Y + p[5]

    # warped image frame @ t+1
    It1x_warp = It1_spline.ev(warpY, warpX)
    # template image fram @ t
    Itx = It_spline.ev(Y, X)

    # f = (It1x_warp - Itx).flatten()
    f = np.linalg.norm(It1x_warp - Itx)
    return f

def bfgsLucasKanade(It, It1, rect, threshold, num_iters):
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
    
    dp = np.array([float("inf")]* 6)
    
    count = 0

    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)

    # warpX = p[0] * X + p[1] * Y + p[2]
    # warpY = p[3] * X + p[4] * Y + p[5]

    # # warped image frame @ t+1
    # It1x_warp = It1_spline.ev(warpY, warpX)
    # # template image fram @ t
    # Itx = It_spline.ev(Y, X)

    # # f = It1x_warp - Itx
    f = obj_fun(p, X, Y, It_spline, It1_spline)
    # print(p.shape)
    # print(f.shape)

    res = minimize(obj_fun, p, args=(X, Y, It_spline, It1_spline), method='BFGS')

    p = res.x
    # print("Result: ", p)
    
    # while np.linalg.norm(dp)>= threshold and count <= num_iters:
        
    #     count += 1

    #     Itx = It_spline.ev(Y, X)
        
    #     warpX = p[0] * X + p[1] * Y + p[2]
    #     warpY = p[3] * X + p[4] * Y + p[5]
        
    #     dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
    #     dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
    #     It1x_warp = It1_spline.ev(warpY, warpX)
        
    #     G = np.array([dwx*X.flatten(),
    #                   dwx*Y.flatten(),
    #                   dwx,
    #                   dwy*X.flatten(),
    #                   dwy*Y.flatten(),
    #                   dwy]).T
        
    #     b = (Itx - It1x_warp).flatten()
            
    #     H = G.T @ G
    #     dp = (np.linalg.inv(H)) @ (G.T @ b)

    #     p+=dp
        
    # M = np.vstack((p.reshape(2,3), M[2,0:3]))

    return p
