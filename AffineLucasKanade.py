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
    alpha = 0.005

    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        
        count += 1

        Itx = It_spline.ev(Y, X)
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        # G_I = [dwx,dwy]
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()

        # Hessian H_I
        H_I = Hessian(It_spline.ev,warpY, warpX,1e-13)

        # dw/dp = [[X,Y,1,0,0,0],[0,0,0,X,Y,1]]
        D_W = np.zeros((2,6,X.shape[0]*X.shape[1]))
        D_W[0,0,:] = X.flatten()
        D_W[1,3,:] = X.flatten()
        D_W[0,1,:] = Y.flatten()
        D_W[1,4,:] = Y.flatten()
        D_W[0,2,:] = np.ones((X.shape[0]*X.shape[1]))
        D_W[1,5,:] = np.ones((X.shape[0]*X.shape[1]))
        D_W = np.transpose(D_W,(2,0,1)) #(num,2,6)
        
        It1x_warp = It1_spline.ev(warpY, warpX)
        
        # set up optimization problem
        A = np.array([dwx*X.flatten(),
                      dwx*Y.flatten(),
                      dwx,
                      dwy*X.flatten(),
                      dwy*Y.flatten(),
                      dwy]).T
        b = (Itx - It1x_warp).flatten()

        # cal gradient
        G_F = -A.T@b

        # cal hessian
        H_F1 = A.T@A # (6,6)
        H_F2 = (H_I@D_W) # (num,2,2)*(num,2,6)
        H_F2 = np.transpose(H_F2,(0,2,1)) #(num,6,2)
        H_F2 = np.sum(H_F2@D_W,axis=0) # (6,6)
        H_F = H_F1 + H_F2

        # update
        dp = -alpha*np.linalg.inv(H_F)@G_F
        p+=dp
        error = np.linalg.norm(b)

    M = np.vstack((p.reshape(2,3), M[2,0:3]))
    print(p,error)
    return p

def Hessian(f,y,x,eps):
    ''' 
        Inputs: It_spline.ev
    '''
    # # calculate first order derivatives w.r.t x and y
    # fy1y = f(y-eps,x,dx=1,dy=0)
    # fy2y = f(y+eps,x,dx=1,dy=0)
    # fx1x = f(y,x-eps,dx=0,dy=1)
    # fx2x = f(y,x+eps,dx=0,dy=1)
    # fx1y = f(y,x-eps,dx=1,dy=0)
    # fx2y = f(y,x+eps,dx=1,dy=0)

    # # calculate 2nd order derivatives: [[fxx,fxy],[fyx,fyy]]
    # fxx = (fx2x-fx1x)/(2*eps)
    # fyy = (fy2y-fy1y)/(2*eps)
    # fxy = (fx2y-fx1y)/(2*eps)

    fxx = f(y,x,dx=0,dy=2)
    fyy = f(y,x,dx=2,dy=0)
    fxy = f(y,x,dx=1,dy=1)
    H = np.array([[fxx,fxy],[fxy,fyy]])
    H = H.reshape((2,2,-1))
    H = H.transpose((2,0,1))

    return H
    