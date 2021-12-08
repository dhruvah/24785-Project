import cv2
import os

def generate_video(image_folder, video_name): 
    # video_name = 'coke_LM.avi'
      
    images = [img for img in os.listdir(image_folder)] 
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    # setting the frame width, height width the width, height of first image 
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, 0, 10, (width, height))  
  
    # Appending the images to the video one by one 
    for i in range(len(images)):  
        video.write(cv2.imread(os.path.join(image_folder, 'coke_tracking%02d.jpg' % i)))  
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated


def compute_iou(gt_box,b_box):
    '''
    :param gt_box: ground truth gt_box = [x0,y0,x1,y1]
    :param b_box: bounding box b_box
    :return: 
    '''
    width0 = gt_box[2] -gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = width0 + width1 -(max_x - min_x)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = height0 + height1 - (max_y - min_y)
 
    interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

