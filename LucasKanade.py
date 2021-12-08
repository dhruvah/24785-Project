import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import optimize

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0
    
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf"), float("inf")])
    
    count = 0
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    x, y = np.meshgrid(x,y)
    # alpha = 0.1 # Gradient Descent
    # alpha = 0.5 # for using H_I as H_F, Newton
    # alpha = 0.8 # for using U_F directly, Newton
    alpha = 1e-6
    
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        count += 1
        
        Itx = It_spline.ev(y, x).flatten()
        It1x_warp = It1_spline.ev(y+p[1], x+p[0]).flatten()
        G = np.array([
            It1_spline.ev(y+p[1], x+p[0], dx = 0, dy = 1).flatten(),
            It1_spline.ev(y+p[1], x+p[0], dx = 1, dy = 0).flatten()]).T

        # # Newton
        # # alpha=0.5
        # # here, x means the first axis.
        # G_I = G
        # H_I = Hessian(It_spline.ev,y+p[1], x+p[0],1e-13)
        # G_F = G_I.T@(It1x_warp-Itx)
        # ## Use H_F or H_I. needs different value of alpha
        # H_F = G_I.T@G_I + np.transpose(H_I,(1,2,0))@(It1x_warp-Itx)
        # # H_I = np.square(np.linalg.norm(H_I,axis=0))
        # dp = -alpha*np.linalg.inv(H_F)@G_F

        # Gauss-Newton
        # alpha = 10
        # also remember to change the threshold to be 1e3
        # H = G.T @ G
        # dp = np.linalg.inv(H) @ G.T @ (Itx - It1x_warp) 

        # Gradient Descent
        ## line search before GD
        # import pdb;pdb.set_trace()
        # result = optimize.line_search(obj_fun, obj_grad, p, G.T @ (Itx - It1x_warp), args=(x, y, It_spline, It1_spline))
        # alpha = result[0]
        # import pdb;pdb.set_trace()
        sk = G.T @ (Itx - It1x_warp)
        # dp = alpha * sk/np.linalg.norm(sk)
        dp = alpha*sk

        ###########tests
        error = np.linalg.norm(Itx - It1x_warp)
        # print("error",error,"norm",np.linalg.norm(dp),"iter",count)

        p+=dp
    print(p,error**2)
    
    
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

def obj_fun(p, x, y, It_spline, It1_spline):
    Itx = It_spline.ev(y, x).flatten()
    It1x_warp = It1_spline.ev(y+p[1], x+p[0]).flatten()
    # f = (It1x_warp - Itx).flatten()
    f = np.linalg.norm(It1x_warp - Itx)
    return f
def obj_grad(p, x, y, It_spline, It1_spline):
    G = np.array([
            It1_spline.ev(y+p[1], x+p[0], dx = 0, dy = 1).flatten(),
            It1_spline.ev(y+p[1], x+p[0], dx = 1, dy = 0).flatten()]).T
    Itx = It_spline.ev(y, x).flatten()
    It1x_warp = It1_spline.ev(y+p[1], x+p[0]).flatten()
    
    return G.T @ (Itx - It1x_warp)
