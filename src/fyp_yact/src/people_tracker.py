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

        self.dequeDetections = deque(maxlen = 2)
        
        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 1)

    #region ROS
    def callback(self, msg):

        img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lstDetections = msg.bounding_boxes

        lstDets = []

        lstHeads = []

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

            
        #region DEEPSORT
        features = self.encoder(img, lstDets)

        # Create DeepSort detections
        trackedObjects = [Detection(bbox, 1.0, feature) for bbox, feature in zip(lstDets, features)]

        self.tracker.predict()
        self.tracker.update(trackedObjects)
        #endregion

        """
        #region TRAJECTORYESTIMATION
        if(self.frameCount % 1 == 0):
            
            xPast = []
            yPast = []
            
            self.dequeDetections.append(self.tracker.tracks)

            for trackCur in self.tracker.tracks:

                for ind in range(len(self.dequeDetections) - 2, -1, -1):
                    # Loop over list of tracks 5, 10, 15 ... frames in the past

                    for trackPast in self.dequeDetections[ind]: 
                        # Find ID if possible
                        if trackPast.track_id == trackCur.track_id:
                            # Get centroids
                            xPast.append(trackPast.mean[0])
                            yPast.append(trackPast.mean[1])
                
                xPast = np.array(xPast)
                yPast = np.array(yPast)

                z = np.polyfit(xPast, yPast, 1)
                p = np.poly1d(z)



                xLeft  = int(trackCur.mean[0]) - 10
                xRight = int(trackCur.mean[0]) + 10

                # pdb.set_trace()


                pos1 = (int(xLeft), int(p(xLeft)))
                pos2 = (int(xRight), int(p(xRight)))


                cv2.line(img, pos1,pos2, (0, 255, 0), 2)

                # Reset histories for next tracker
                xPast = []
                yPast = []
        #endregion
        """


        if self.debug > 0:

            #FPS Calculation (Measure time for 10 frames)
            if(self.frameCount % 10 == 0):
                timer         = time.time() - self.timePrev
                self.timePrev = time.time()

                self.intFPS   = int(10/timer)

            self.displayDetections(img)

        self.frameCount += 1

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

            # Print FPS
            cv2.putText(img, 'FPS: {}'.format(self.intFPS), (img.shape[1] - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

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