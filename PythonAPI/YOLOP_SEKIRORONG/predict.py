#!/usr/bin/env python

#-----------------------------------------------------------------------#
#   predict.py integrates functions such as single image prediction, camera detection, FPS testing, and directory traversal detection into one file. 
#   The mode can be modified by specifying the mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

from config import depth_img

rgb_path = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/Cam_Recorder/val'
depth_path = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/val'

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the test mode:
    #   'predict' means single image prediction. If you want to modify the prediction process, such as saving images or capturing objects, you can first look at the detailed comments below.
    #   'video' means video detection, you can call the camera or video for detection, see the comments below for details.
    #   'fps' means testing fps, the image used is street.jpg in img, see the comments below for details.
    #   'dir_predict' means traversing the folder for detection and saving. By default, it traverses the img folder and saves the img_out folder, see the comments below for details.
    #----------------------------------------------------------------------------------------------------------#
    if depth_img:
        mode = "dir_predict"
    else:
        mode = "video"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path is used to specify the path of the video. When video_path=0, it means detecting the camera.
    #   To detect video, set it to video_path = "xxx.mp4", which means reading the xxx.mp4 file in the root directory.
    #   video_save_path means the path to save the video. When video_save_path="", it means not to save.
    #   To save the video, set it to video_save_path = "yyy.mp4", which means saving it as the yyy.mp4 file in the root directory.
    #   video_fps is the fps of the saved video.
    #   video_path, video_save_path, and video_fps are only valid when mode='video'.
    #   To save the video, you need to exit with ctrl+c or run to the last frame to complete the full saving steps.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/videos/val_3.mp4'
    video_save_path = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder/videos/validation_output.mp4'
    video_fps       = 5.0
    #-------------------------------------------------------------------------#
    #   test_interval is used to specify the number of times the image is detected when measuring fps.
    #   In theory, the larger the test_interval, the more accurate the fps.
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path specifies the folder path for the images to be detected.
    #   dir_save_path specifies the save path for the detected images.
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/Cam_Recorder/val"
    dir_save_path   = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/Cam_Recorder/val_output"

    if mode == "predict":
        '''
        1. If you want to save the detected image, you can save it using r_image.save("img.jpg"), and modify it directly in predict.py.
        2. If you want to get the coordinates of the prediction box, you can go into the yolo.detect_image function and read the values of top, left, bottom, right in the drawing part.
        3. If you want to crop the target using the prediction box, you can go into the yolo.detect_image function and use the top, left, bottom, right values obtained in the drawing part 
           to crop the original image using matrix operations.
        4. If you want to write extra text on the predicted image, such as the number of detected specific targets, you can go into the yolo.detect_image function and judge the predicted_class 
           in the drawing part. For example, judge if predicted_class == 'car': to determine whether the current target is a car, and then record the quantity. You can write text using draw.text.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video) correctly. Please check whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while True:
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Format conversion, BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform detection
            result = yolo.detect_image(frame)
            fps = (fps + (1. / (time.time() - t1))) / 2
            newframe = np.array(result)
            # RGB to BGR to meet opencv display format
            newframe = cv2.cvtColor(newframe, cv2.COLOR_RGB2BGR)
            
            # fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f" % (fps))
            new_frame = cv2.putText(newframe, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video", newframe)
            c = cv2.waitKey(1) & 0xff 
            if video_save_path != "":
                out.write(new_frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                if depth_img:
                    image_path = image_path.strip(rgb_path)
                    image_path = depth_path + '/' + image_path
                    d_image = Image.open(image_path)
                    r_image = yolo.detect_image(image, d_image)
                else:
                    r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                # print(r_image.shape)
                img = Image.fromarray(r_image)
                # print(img.shape)
                save_path = os.path.join(dir_save_path, img_name)
                img.save(save_path)
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
