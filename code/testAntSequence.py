import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.05, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

num_frames = seq.shape[2]

for frame in range(num_frames - 1):
    image1 = seq[:,:,frame]
    image2 = seq[:,:,frame + 1]
    
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    ants = np.where(mask == 0)
    
    if frame == 30 or frame == 60 or frame == 90 or frame == 120:
        plt.figure()
        plt.axis('off')
        plt.imshow(image2, cmap = 'gray')
        plt.plot(ants[1], ants[0],'.',color = 'r')
        plt.show()