import argparse
import numpy as np
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


seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]


carseqrects = np.array(rect)
num_frames = seq.shape[2]


for frame in range(num_frames - 1):
    
    It = seq[:,:,frame]
    It1 = seq[:,:,frame+1]
    
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    
    rect = rect + np.array([p[0], p[1], p[0], p[1]])
    carseqrects = np.vstack((carseqrects, rect))
    
    if frame ==1 or frame ==100 or frame == 200 or frame == 300 or frame == 400:
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(It1, cmap = 'gray')
        patch = patches.Rectangle((rect[0],  rect[1]),
                                  (rect[2] - rect[0]),
                                  (rect[3] - rect[1]),
                                  linewidth = 1, edgecolor = 'r',
                                  facecolor = 'none')
        ax.add_patch(patch)
        plt.show()

np.save("../result/carseqrects.npy",carseqrects)