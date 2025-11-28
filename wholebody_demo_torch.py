import time

import cv2
import torch
import numpy as np

from rtmlib_torch import PoseTracker, Wholebody, draw_skeleton
from utils_inference.video.video_data import VideoData
from utils_inference.video.video_info_ffmpeg import VideoInfoFfmpeg
from utils_inference.video.video_writer import TensorSaveVideo

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
device = 'cuda:0'
backend = 'tensorrt'  # opencv, onnxruntime, openvino

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = PoseTracker(
    Wholebody,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device)

reader = VideoInfoFfmpeg('./data/test_data/a121f56d9bed4d7b29db349559eb7897.mp4')
cap = cv2.VideoCapture('./data/test_data/a121f56d9bed4d7b29db349559eb7897.mp4')
batch_size = 1
pts = list(list(range(index, index + batch_size if index + batch_size < reader.num_frame else reader.num_frame)) for index in range(0, reader.num_frame, batch_size))

frame_idx = 0

for i in pts:
    #ffmpeg
    # frame = reader.load_data_by_index(i, device = 'cpu')[..., [2, 1, 0]]  # RGB to BGR
    # wholebody(frame.data)

    # cv2 编解码一致，用于结果对齐
    frame_cv_batch = []
    for _ in i:
         success, frame_cv = cap.read()
         frame_cv_batch.append(frame_cv)
    frame_cv_torch = np.array(frame_cv_batch)
    frame = torch.from_numpy(frame_cv_torch).to(device=device)
    keypoints, scores = wholebody(frame)
    
    id = 0
    for _ in i:
        frame_idx += 1
        img_show = frame_cv_batch[id].copy()
        id += 1

        img_show = draw_skeleton(img_show,
                             keypoints[id - 1].cpu().numpy(),
                             scores[id - 1].cpu().numpy(),
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=4)
    
        cv2.imwrite(f'./data/result_torch_trt_{batch_size}/{frame_idx:05d}.jpg', img_show)

