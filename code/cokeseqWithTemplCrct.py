import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

# seq = np.load("../data/carseq.npy")
rect = [298, 160, 346, 240]
img_path = glob.glob("../dataset/coke/*.jpg")
img_sorted = sorted([x for x in img_path])

rect0 = rect
p0 = np.zeros(2)
cokerects_wcrt = np.array(rect)
num_frames = len(img_sorted)

template_threshold = -1

im0 = img_sorted[0]
T1x = cv2.imread(str(im0), cv2.IMREAD_GRAYSCALE)
It = cv2.imread(str(im0), cv2.IMREAD_GRAYSCALE)
i = 0
for frame in range(num_frames - 1):
    
    im1 = img_sorted[frame+1]
    It1 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE)
    
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0)
    p_n = p + [rect[0] - rect0[0], rect[1] - rect0[1]]
    
    pstar_n = LucasKanade(T1x, It1, rect0, threshold, num_iters, p_n)
    print(p_n, pstar_n)
    
    # print(np.linalg.norm(p_n - pstar_n))
    if np.linalg.norm(p_n - pstar_n) < template_threshold:
        pstar = pstar_n - [rect[0] - rect0[0], rect[1] - rect0[1]]
        
        rect = rect + np.array([pstar[0], pstar[1], pstar[0], pstar[1]])
        cokerects_wcrt = np.vstack((cokerects_wcrt, rect))
        im2 = img_sorted[frame+1]
        It = cv2.imread(str(im2), cv2.IMREAD_GRAYSCALE)
        p0 = np.zeros(2)
        
    else:
        rect = rect + np.array([p[0], p[1], p[0], p[1]])
        cokerects_wcrt = np.vstack((cokerects_wcrt, rect))
        p0 = p
    
    # plt.savefig("../dataset/coke_crct_results/coke_tracking%02d.jpg" % i)
    i = i + 1

# frameindex = [1, 100, 200, 300, 400]    
cokeseqrects = np.load("../result/cokeseqrects.npy")
# img_path = glob.glob("../dataset/coke_results/coke_tracking*.jpg")
# img_sorted = sorted([x for x in img_path])


for i in range(num_frames - 1):
    rect = cokeseqrects[i,:]
    rect_wcrt = cokerects_wcrt[i,:]
    

    fig, ax = plt.subplots()
    plt.axis('off')
    im3 = img_sorted[i]
    It3 = cv2.imread(str(im3), cv2.IMREAD_GRAYSCALE)
    ax.imshow(It3, cmap = 'gray')
    patch1 = patches.Rectangle((rect[0],  rect[1]),
                               (rect[2] - rect[0]),
                               (rect[3] - rect[1]),
                               linewidth = 1, edgecolor = 'r',
                               facecolor = 'none')
    
    patch2 = patches.Rectangle((rect_wcrt[0],  rect_wcrt[1]),
                               (rect_wcrt[2] - rect_wcrt[0]),
                               (rect_wcrt[3] - rect_wcrt[1]),
                               linewidth = 1, edgecolor = 'b',
                               facecolor = 'none')
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    # plt.show()
    plt.savefig("../dataset/coke_crct_results/coke_tracking%02d.jpg" % i)

np.save("../result/cokeseqrects-wcrt.npy",cokerects_wcrt)