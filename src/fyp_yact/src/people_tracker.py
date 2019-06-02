#!/usr/bin/env python

import sys, os

from deep_sort.deep_sort           import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker   import Tracker
from deep_sort.tools               import generate_detections as gdet

from fyp_yact.msg import BoundingBox, CompressedImageAndBoundingBoxes # Input Messages
from fyp_yact.msg import BoundingBoxID, DetectionAndID  # Output Messages

from   collections import deque
import cv2
import numpy as np

import cv_bridge as bridge
import rospy
from   sensor_msgs.msg import CompressedImage

import time

import pdb


class yact_node:

    def __init__(self, debug = 0):
        
        self.debug = debug
        
        self.frameCount = 0
        self.timePrev   = time.time()

        #region DEEPSORT
        metric = nn_matching.NearestNeighborDistanceMetric('cosine',
                                                           matching_threshold = 0.2,
                                                           budget = 100)
        self.tracker = Tracker(metric)

        absFilePath = os.path.abspath(__file__)
        fileDir     = os.path.dirname(absFilePath)

        filePathModel = os.path.join(fileDir, 'deep_sort/resources/networks/mars-small128.pb')
        self.encoder  = gdet.create_box_encoder(filePathModel, batch_size=1)
        #endregion

        self.detectionAndId = []

        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 1)

        self.publisherDetectionID      = rospy.Publisher('yact/output/detectionid',
                                                            DetectionAndID,
                                                            queue_size = 1)


    def callback(self, msg):

        self.img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)

        lstDetections = msg.bounding_boxes

        lstDetsDeepSort = []

        if( lstDetections ):
            
            for det in lstDetections:

                if det.Class == 'person':
                    
                    # Deep Sort Bounding Boxes
                    # Use the format TOP LEFT WIDTH HEIGHT (tlwh)
                    dsWidth  = det.xmax - det.xmin
                    dsHeight = det.ymax - det.ymin 
                    lstDetsDeepSort.append([det.xmin, det.ymin, dsWidth, dsHeight])

        self.detectionAndId = []

        #region DEEPSORT
        if self.frameCount % 3 == 0:
            features = self.encoder(self.img, lstDetsDeepSort)
    
            # Create DeepSort detections
            trackedObjects = [Detection(bbox, 1.0, feature) for bbox, feature in zip(lstDetsDeepSort, features)]
    
            self.tracker.predict()
            self.tracker.update(trackedObjects)
        #endregion

        self.displayDetections()

        msgDetectionAndID            = DetectionAndID()
        msgDetectionAndID.header     = msg.header
        msgDetectionAndID.detections = self.detectionAndId

        #region DISPLAY
        if self.frameCount % 10 == 0:
            timer         = time.time() - self.timePrev
            self.timePrev = time.time()
            self.intFPS   = int(10/timer)

        # Print FPS
        cv2.putText(self.img, 'FPS: {}'.format(self.intFPS), (self.img.shape[1] - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow('yact: people_tracker.py', self.img)

        cv2.waitKey(3)

        self.frameCount += 1
        #endregion


    #region DEBUG 
    def displayDetections(self):

        for track in self.tracker.tracks:
            
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()

            # Construct message
            msgBbid = BoundingBoxID()
            msgBbid.boundingBox.xmin = int(bbox[0])
            msgBbid.boundingBox.ymin = int(bbox[1])
            msgBbid.boundingBox.xmax = int(bbox[2])
            msgBbid.boundingBox.ymax = int(bbox[3])
            msgBbid.id = int(track.track_id)

            self.detectionAndId.append(msgBbid)

            # Draw bounding box and label
            cv2.rectangle(self.img, (int(bbox[0]) , int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            labelText = 'ID: {}'.format(track.track_id)
            cv2.putText(self.img, labelText, (int(bbox[0]), int(bbox[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    #endregion


def main(args):
    rospy.init_node('yact_node', anonymous=True)
    yn = yact_node(debug = 1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down YACT node')
    
if __name__ == '__main__':
    main(sys.argv)