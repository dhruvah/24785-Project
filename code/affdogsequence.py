import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from AffineLucasKanade import AffineLucasKanade

from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold


# seq = np.load("../data/carseq.npy")
img_path = glob.glob("../dataset/dog1/*.png")
img_sorted = sorted([x for x in img_path])
# img = img_sorted[50]
# img1 = cv2.imread(str(img))
# cv2.imshow("Output", img1)
# cv2.waitKey(0)
rect = [139,113,191,186]

dogseqrects = np.array(rect)
# # num_frames = seq.shape[2]
# i = 0
# print("Hello!")

for i in range(len(img_sorted)-1):
    
    # print(i)
    im = img_sorted[i]
    It = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
    im1 = img_sorted[i+1]
    It1 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE)
        
    p = AffineLucasKanade(It, It1, rect, threshold, num_iters)
    
    rect = np.array([p[0] * rect[0] + p[1] * rect[1] + p[2], p[3] * rect[0] + p[4] * rect[1] + p[5], 
                    p[0] * rect[2] + p[1] * rect[3] + p[2], p[3] * rect[2] + p[4] * rect[3] + p[5]])
    dogseqrects = np.vstack((dogseqrects, rect))
    
    # if frame ==1 or frame ==100 or frame == 200 or frame == 300 or frame == 400:
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(It1, cmap = 'gray')
    patch = patches.Rectangle((rect[0],  rect[1]),
                                (rect[2] - rect[0]),
                                (rect[3] - rect[1]),
                                linewidth = 1, edgecolor = 'r',
                                facecolor = 'none')
    ax.add_patch(patch)
    # plt.show()
    plt.savefig("../dataset/dog_results_affine/dog_tracking%02d.jpg" % i)

np.save("../result/affinedogseqrects.npy",dogseqrects)