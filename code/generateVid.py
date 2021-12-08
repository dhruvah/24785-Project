import cv2
import os

def generate_video(): 
    # image_folder = '../dataset/coke_results/' # make sure to use your folder 
    image_folder = '../dataset/bike_results_affine/' # make sure to use your folder
    video_name = 'biker_bfgs.avi'
      
    images = [img for img in os.listdir(image_folder)] 
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    # setting the frame width, height width the width, height of first image 
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, 0, 25, (width, height))  
  
    # Appending the images to the video one by one 
    for i in range(len(images)):  
        video.write(cv2.imread(os.path.join(image_folder, "biker_tracking%02d.jpg" % i))) 
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated
    
generate_video()