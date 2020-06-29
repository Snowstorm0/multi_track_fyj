#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
frame_num = 0
def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.15
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False#   True                        # 172.18.136.43_01_20190709093456346_4_20190712101304
                                                    # 172.18.136.43_01_20190705101620718_3_20190711140736_20190711140913
                                                    # 172.18.136.43_01_20190708133642769_20190711144335
    video_capture = cv2.VideoCapture('./hard_01.mp4') 

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output_temp.avi', fourcc, 15, (w, h))
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        global frame_num
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb？  #fromarray 从array到image的转换   
        boxs = yolo.detect_image(image)  # 
        print('>>>>>>----------------------{}----------------------<<<<<<'.format(frame_num))
        # # print("box_num",len(boxs))
        features = encoder(frame,boxs) #encoder : Callable[image, ndarray] -> ndarray  The encoder function takes as input a BGR color image and a matrix of bounding boxes in format `(x, y, w, h)` and returns a matrix of corresponding feature vectors.
        
        # score to 1.0 here). detection.py
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]   #detections 转为 tlwh, confidence, feature
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        frame_num = + frame_num +1

        # if(len(detections)!=0):
            # global frame_num
            # frame_num = + frame_num +1
            # f_feature = open('E:/project/multi_track/deep_sort/tmp/02/features/{:05d}.txt'.format(frame_num),'a')
            # f_feature.write(str(features)+'\n')
            # print(features)

        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0), 2)
            # im_tmp = cv2.getRectSubPix(frame,(int(bbox[2]), int(bbox[3])),(int(bbox[0]), int(bbox[1])))
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)
        cv2.waitKey(1)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            # print("box_num",len(boxs))
            # if len(boxs) != 0:
            #     for i in range(0,len(boxs)):
            #         print("box_num",len(boxs))
            
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
