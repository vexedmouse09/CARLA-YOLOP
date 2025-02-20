#!/usr/bin/env python
# @Time : 2021/12/28 18:07
# @Author : 戎昱
# @File : config.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import os

# ---------------------Generator Parameters------------------------
status = 'train' # train/test/val

round = '5' # Avoiding the generated data conflicting

carla_map = 'Town05' # Town01,02,03,04,05,10HD available  Town03有上下坡！ Recommend Town05

weather_index = 0 # change the weather when simulating

Ratio_of_vehicles = 0.1 # Ratio of vehicles/walkers to spawn points
Ratio_of_walkers = 0.5 # Ratio_of_vehicles + Ratio_of_walkers is recommended to less than 0.8
# ------------------------Path----------------------------------------

kitti = True
kitti_root = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder'

if status == 'test':
    kitti_training = kitti_root + r'/testing'
else:
    kitti_training = kitti_root + r'/training'

# kitti_training = kitti_root + r'\testing'

Carla_Recorder_dir = r'/home/reu/carla/PythonAPI/examples/Carla_Recorder'

txt_path = Carla_Recorder_dir + '/Position_Recorder' + '/' + status
rgb_path = Carla_Recorder_dir + '/Cam_Recorder' + '/' + status
semantic_path = Carla_Recorder_dir + '/semantic_Recorder' + '/' + status
lidar_path = Carla_Recorder_dir + '/Lidar_Recorder' + '/' + status
depth_path = Carla_Recorder_dir + '/Depth_Recorder' + '/' + status
semantic2_path = Carla_Recorder_dir + '/semantic2_Recorder' + '/' + status
laneline_file = r"/Laneline_Recorder" + '/' + status # Deprecated
save_path = Carla_Recorder_dir + r'/videos'
tmp_path = Carla_Recorder_dir + r'/tmp'

recorder_dir = Carla_Recorder_dir + r'/recording01.log'

if kitti:
    rgb_path = kitti_training + r'/image_2'
    lidar_path = kitti_training + r'/velodyne'
    semantic2_path = kitti_training + r'/semantic'
    label_path = kitti_training + r'/label_2'
    paintedVelodyne_path = kitti_training + r'/paintedVelodyne'

# ----deprecated------
save_3dbbox_path = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/3Dbbox" + '/' + status
save_laneline_path = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/LaneLine" + '/' + status
save_drivearea_path = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/DriveableArea" + '/' + status
save_img_path = r"/home/reu/carla/PythonAPI/examples/Carla_Recorder/Image" + '/' + status + "/images.txt"
# --------------------

if kitti:
    save_3dbbox_path = Carla_Recorder_dir + r'/tmp'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

mkdir(rgb_path)
mkdir(lidar_path)
mkdir(semantic2_path)
mkdir(label_path)
mkdir(semantic_path)
mkdir(txt_path)
mkdir(depth_path)
mkdir(save_path)