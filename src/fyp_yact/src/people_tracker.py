#!/usr/bin/env python

import sys, os

from deep_sort.deep_sort           import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker   import Tracker
from deep_sort.tools               import generate_detections as gdet

from fyp_yact.msg import BoundingBox, CompressedImageAndBoundingBoxes

from   collections import deque
import cv2
import numpy as np

import cv_bridge as bridge
import rospy
from   sensor_msgs.msg import CompressedImage

import cProfile, pstats


class yact_node:

    def __init__(self, debug = 0):
        
        self.debug = debug

        self.frameCount = 0
        self.frameSkip  = 1

        # Deep Sort
        metric = nn_matching.NearestNeighborDistanceMetric('cosine',
                                                           matching_threshold = 0.2,
                                                           budget = 100)
        self.tracker        = Tracker(metric)
        self.trackedObjects = []

        filePathModel = '/home/aufar/fyp_yact/src/fyp_yact/src/deep_sort/resources/networks/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(filePathModel, batch_size=1)

        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 1)

    #region ROS
    def callback(self, msg):
        
        img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)

        lstDetections = msg.bounding_boxes

        lstDets = []

        if( lstDetections ):
            
            for det in lstDetections:
                
                if det.Class == 'person':
                    # Deep Sort Bounding Boxes
                    # Use the format TOP LEFT WIDTH HEIGHT (tlwh)
                    dsWidth  = det.xmax - det.xmin
                    dsHeight = det.ymax - det.ymin 
                    lstDets.append([det.xmin, det.ymin, dsWidth, dsHeight])
        else:
            lstDets = []

        if(self.frameCount % self.frameSkip == 0):
            
            #region DEEPSORT
            
            features = self.encoder(img, lstDets)

            # Create DeepSort detections
            self.trackedObjects = [Detection(bbox, 1.0, feature) for bbox, feature in zip(lstDets, features)]
    
            self.tracker.predict()
            self.tracker.update(self.trackedObjects)

            #endregion

        if self.debug > 0:
            # self.estimatePersonMotion(img)
            self.displayDetections(img)

    #endregion


    #region HELPERFNS
    def convertBBToCentroid(self, bbox):
        '''
        arguments:
            bbox: array [xmin, ymin, xmax, ymax]
        
        returns:
            centroid: array [xcentre, ycentre]
        '''

        xcentre = float((bbox[0] + bbox[2]) / 2)
        ycentre = float((bbox[1] + bbox[3]) / 2)

        return np.asarray([xcentre, ycentre]).reshape((2,1))
    #endregion


    #region DEBUG 
    def displayDetections(self, img):

        for track in self.tracker.tracks:
            
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()

            # Draw bounding box and label
            cv2.rectangle(img, (int(bbox[0]) , int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
            labelText = 'ID: {}'.format(track.track_id)
            cv2.putText(img, labelText, (int(bbox[0]), int(bbox[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow('yact: people_tracker.py', img)
        cv2.waitKey(3)
    #endregion


def main(args):
    yn = yact_node(debug = 1)
    rospy.init_node('yact_node', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down YACT node')
    
if __name__ == '__main__':
    main(sys.argv)