import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
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
    
    x1, y1, x2, y2 = 0, 0, It.shape[1] -1, It.shape[0] -1
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf")]* 6)
    
    count = 0
   
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        x = np.arange(x1, x2 + 1)
        y = np.arange(y1, y2 + 1)
        X, Y = np.meshgrid(x,y)
        count += 1
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        commonpts = np.where((warpX >= x1) & (warpX <= x2) & 
                             (warpY >= y1) & (warpY <= y2), True, False) 
        
        X, Y = X[commonpts], Y[commonpts]
        warpX, warpY = warpX[commonpts], warpY[commonpts]
        
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
        It1x_warp = It1_spline.ev(warpY, warpX)
        
        A = np.array([dwx*X,
                      dwx*Y,
                      dwx,
                      dwy*X,
                      dwy*Y,
                      dwy]).T
        
        b = (It[commonpts] - It1x_warp).flatten()
        # print(np.shape(A))
    
        
        # print(np.shape(b))
        
        dp = (np.linalg.inv(A.T @ A)) @ (A.T @ b)

        # H = A.T @ A
        # dp = np.linalg.inv(H) @ A.T @ (It[commonpts] - It1x_warp)
        # print(np.shape(dp))
        p+=dp
        
    M = np.vstack((p.reshape(2,3), M[2,0:3]))

    return M

# import numpy as np
# from scipy.interpolate import RectBivariateSpline

# def LucasKanadeAffine(It, It1, threshold, num_iters):
#     """
#     :param It: template image
#     :param It1: Current image
#     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
#     :param num_iters: number of iterations of the optimization
#     :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
#     """

#     # put your implementation here
#     M = np.eye(3)
#     p = np.zeros(6)
#     x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
#     rows, cols = It.shape
    
#     y = np.arange(0, rows, 1)
#     x = np.arange(0, cols, 1)     
#     c = np.linspace(x1, x2, cols)
#     r = np.linspace(y1, y2, rows)
#     cc, rr = np.meshgrid(c, r)
    
#     Iy, Ix = np.gradient(It1)
    
#     spline = RectBivariateSpline(y, x, It)
#     T = spline.ev(rr, cc)
    
#     spline_gx = RectBivariateSpline(y, x, Ix)
#     spline_gy = RectBivariateSpline(y, x, Iy)
#     spline1 = RectBivariateSpline(y, x, It1)

#     for k in range(0,int(num_iters)):
#         # print(k)
#         W = np.array([[1.0 + p[0], p[1], p[2]],
#                        [p[3], 1.0 + p[4], p[5]]])
    
#         x1_w = W[0,0] * x1 + W[0,1] * y1 + W[0,2]
#         y1_w = W[1,0] * x1 + W[1,1] * y1 + W[1,2]
#         x2_w = W[0,0] * x2 + W[0,1] * y2 + W[0,2]
#         y2_w = W[1,0] * x2 + W[1,1] * y2 + W[1,2]
    
#         cw = np.linspace(x1_w, x2_w, It.shape[1])
#         rw = np.linspace(y1_w, y2_w, It.shape[0])
#         ccw, rrw = np.meshgrid(cw, rw)
        
#         warpImg = spline1.ev(rrw, ccw)

#         #compute error image
#         #errImg is (n,1)
#         err = T - warpImg
#         errImg = err.reshape(-1,1)
        
#         #compute gradient
#         Ix_w = spline_gx.ev(rrw, ccw)
#         Iy_w = spline_gy.ev(rrw, ccw)
#         #I is (n,2)
#         I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
#         #evaluate delta = I @ jac is (n, 6)
#         delta = np.zeros((It.shape[0]*It.shape[1], 6))
   
#         for i in range(It.shape[0]):
#             for j in range(It.shape[1]):
#                 #I is (1,2) for each pixel
#                 #Jacobiani is (2,6)for each pixel
#                 I_indiv = np.array([I[i*It.shape[1]+j]]).reshape(1,2)
                
#                 jac_indiv = np.array([[j, 0, i, 0, 1, 0],
#                                       [0, j, 0, i, 0, 1]]) 
#                 delta[i*It.shape[1]+j] = I_indiv @ jac_indiv
        
#         #compute Hessian Matrix
#         #H is (6,6)
#         H = delta.T @ delta
        
#         #compute dp
#         #dp is (6,6)@(6,n)@(n,1) = (6,1)
#         dp = np.linalg.inv(H) @ (delta.T) @ errImg
        
#         #update parameters
#         p[0] += dp[0,0]
#         p[1] += dp[1,0]
#         p[2] += dp[2,0]
#         p[3] += dp[3,0]
#         p[4] += dp[4,0]
#         p[5] += dp[5,0]
        
#         if np.square(dp).sum() < threshold:
#             break

#     M =  np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]], [0., 0., 1.0]])
#     # print(M.shape)

#     return M