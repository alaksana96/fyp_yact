#!/usr/bin/env python

import sys, os

''' 
    Adding SORT to Python Path
    This is done so we can access the sort.py functions
'''
absFilePath = os.path.abspath(__file__)
fileDir     = os.path.dirname(absFilePath)
sortPath    = os.path.join(fileDir, 'sort')

print('Adding SORT to Python Path by inserting: {}'.format(sortPath))
sys.path.insert(0, sortPath)

import sort

from fyp_yact.msg import BoundingBox, CompressedImageAndBoundingBoxes

import numpy as np
import cv2

import cv_bridge as bridge
import rospy
from   sensor_msgs.msg import CompressedImage


class yact_node:

    def __init__(self, debug = 0):
        
        self.debug = debug

        self.sort = sort.Sort()

        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 10)


    def callback(self, msg):

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

        self.trackers = self.sort.update(npDets)


        if self.debug > 0:
            img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)
            self.displayDetections(self.trackers, img)



    def displayDetections(self, detections, img):

        for detected in detections:
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


def main(args):
    yn = yact_node(debug = 1)
    rospy.init_node('yact_node', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down YACT node')
    
if __name__ == '__main__':
    main(sys.argv)