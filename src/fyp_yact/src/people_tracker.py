#!/usr/bin/env python

import sys, os

''' 
    Adding SORT to Python Path
'''
absFilePath = os.path.abspath(__file__)
fileDir     = os.path.dirname(absFilePath)
sortPath    = os.path.join(fileDir, 'sort')

print('Adding SORT to Python Path by inserting: {}'.format(sortPath))
sys.path.insert(0, sortPath)

import sort

'''
Deep Sort Imports
'''
from deep_sort.deep_sort           import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker   import Tracker

from fyp_yact.msg import BoundingBox, CompressedImageAndBoundingBoxes

from   collections import deque
import cv2
import numpy as np

import cv_bridge as bridge
import rospy
from   sensor_msgs.msg import CompressedImage


class yact_node:

    def __init__(self, debug = 0):
        
        self.debug = debug

        self.sort = sort.Sort()

        # Deep Sort
        metric = nn_matching.NearestNeighborDistanceMetric('cosine',
                                                           matching_threshold = 0.2,
                                                           budget = 100)
        self.tracker        = Tracker(metric)
        self.trackedObjects = []

        self.dequeDetections = deque(maxlen = 10)

        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 10)

    #region ROS
    def callback(self, msg):

        img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)

        lstDetections = msg.bounding_boxes

        lstDets = []

        if( lstDetections ):
            
            for det in lstDetections:
                
                if det.Class == 'person':
                    lstDets.append([det.xmin, 
                                    det.ymin, 
                                    det.xmax, 
                                    det.ymax, 
                                    det.probability])
        else:
            lstDets = []

        npDets = np.array(lstDets)

        self.trackersCur  = self.sort.update(npDets)

        # Update deque
        zDetections = []

        for tracker in self.trackersCur:
            z = {'ID' : tracker[4], 'zbox' : self.convertBBToCentroid(tracker[0:4])}
            zDetections.append(z)

        self.dequeDetections.append(zDetections)

        # Predict direction of motion for objects
        

        if self.debug > 0:
            # self.estimatePersonMotion(img)
            self.displayDetections(img)
    #endregion


    #region MOTIONESTIMATOR
    def estimatePersonMotion(self, img):

        trackerCurrentCentroids = self.dequeDetections[-1]

        for tracker in trackerCurrentCentroids:
    
            trackerX = []
            trackerY = []

            for ind in range(len(self.dequeDetections) - 2, -1, -1):
            
                for zDetection in self.dequeDetections[ind]:

                    if(zDetection['ID'] == tracker['ID']):

                        trackerX.append(np.asscalar(zDetection['zbox'][0]))
                        trackerY.append(np.asscalar(zDetection['zbox'][1]))

            trackerZ = np.polyfit(np.array(trackerX), np.array(trackerY), 4)
            trackerF = np.poly1d(trackerZ)

            cv2.line(img, 
                     (int(np.asscalar(tracker['zbox'][0]) - 5), int(np.asscalar(tracker['zbox'][1] -5 ))),
                     (int(np.asscalar(tracker['zbox'][0]) + 5), int(trackerF(np.asscalar(tracker['zbox'][0]) + 5))),    
                     (0,255,0),
                     5)
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

        for detected in self.trackersCur:
            # x,y co-ordinates are the centre of the object
            xmin = int(detected[0])
            ymin = int(detected[1])

            xmax = int(detected[2])
            ymax = int(detected[3])

            # Draw bounding box and label
            cv2.rectangle(img, (xmin , ymin), (xmax, ymax), (255,0,0))
            labelText = 'ID: {}'.format(detected[4])
            cv2.putText(img, labelText, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

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