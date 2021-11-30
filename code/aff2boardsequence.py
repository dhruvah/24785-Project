import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=5, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=150, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

# seq = np.load('../data/antseq.npy')
img_path = glob.glob("../dataset/board/*.jpg")
img_sorted = sorted([x for x in img_path])

# num_frames = seq.shape[2]

for i in range(len(img_sorted) - 1):
    # image1 = seq[:,:,i]
    # image2 = seq[:,:,i + 1]
    im = img_sorted[i]
    image1 = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
    im1 = img_sorted[i+1]
    image2 = cv2.imread(str(im1), cv2.IMREAD_GRAYSCALE)
    
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    ants = np.where(mask == 0)
    
    # if i == 30 or i == 60 or i == 90 or i == 120:
    plt.figure()
    plt.axis('off')
    plt.imshow(image2, cmap = 'gray')
    plt.plot(ants[1], ants[0],'.',color = 'r', markersize=2)
    # plt.show()
    plt.savefig("../dataset/board_results_test/board_tracking%02d.jpg" % i)