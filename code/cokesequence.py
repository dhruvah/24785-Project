import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util import *

from LucasKanade import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold


# seq = np.load("../data/carseq.npy")
img_path = glob.glob("../data/coke/*.jpg")
img_sorted = sorted([x for x in img_path])
# img = img_sorted[50]
# img1 = cv2.imread(str(img))
# cv2.imshow("Output", img1)
# cv2.waitKey(0)
rect = [298, 160, 346, 240]

cokeseqrects = np.array(rect)
# # num_frames = seq.shape[2]
# i = 0
# print("Hello!")

gt = np.loadtxt("../data/coke/gt.txt", delimiter=',')
iou_list = [1]
n = len(img_sorted) - 1
for i in range(n):
    
    print(i)
    im = img_sorted[i]
    It = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
    im1 = img_sorted[i+1]
    It1 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE)
        
    p = LucasKanade_LM(It, It1, rect, threshold, num_iters)
    
    rect = rect + np.array([p[0], p[1], p[0], p[1]])
    cokeseqrects = np.vstack((cokeseqrects, rect))
    
    # # if frame ==1 or frame ==100 or frame == 200 or frame == 300 or frame == 400:
    # fig, ax = plt.subplots()
    # plt.axis('off')
    # ax.imshow(It1, cmap = 'gray')
    # patch = patches.Rectangle((rect[0],  rect[1]),
    #                             (rect[2] - rect[0]),
    #                             (rect[3] - rect[1]),
    #                             linewidth = 1, edgecolor = 'r',
    #                             facecolor = 'none')
    # ax.add_patch(patch)
    # # plt.show()
    # plt.savefig("../data/coke_results/coke_tracking%02d.jpg" % i)
    # plt.close(fig)

    iou_list.append(compute_iou(gt[i], rect))

np.save("../result/cokeseqrects.npy",cokeseqrects)
plt.plot(np.arange(n + 1), iou_list)
plt.xlabel("Frame")
plt.ylabel('IOU')
plt.show()