#!/usr/bin/env python

import sys, os

from openpose import pyopenpose as op


from fyp_yact.msg import BoundingBox, CompressedImageAndBoundingBoxes # Input Messages
from fyp_yact.msg import BoundingBoxDirection, DetectionAndDirection  # Output Messages

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

        #region OPENPOSE
        opParams = dict()
        opParams['model_folder']   = '/home/aufar/Documents/openpose/models/'
        opParams['net_resolution'] = '176x176'

        self.opWrapper = op.WrapperPython()
        
        self.opWrapper.configure(opParams)
        self.opWrapper.start()

        self.datum = op.Datum()
        #endregion

        self.subscriberImageDetections = rospy.Subscriber('yolo_detector/output/compresseddetections',
                                                          CompressedImageAndBoundingBoxes,
                                                          self.callback,
                                                          queue_size = 1)

        self.publisherDetectionDirections = rospy.Publisher('people_tracker/output/detectiondirections',
                                                            DetectionAndDirection,
                                                            queue_size = 1)


    def callback(self, msg):

        self.img = cv2.imdecode(np.fromstring(msg.data, np.uint8), 1)

        lstDetections = msg.bounding_boxes

        lstDets = []

        if( lstDetections ):
            
            for det in lstDetections:

                if det.Class == 'person':
                    lstDets.append([det.xmin, det.ymin, det.xmax, det.ymax])


        self.datum.cvInputData = self.img
        self.opWrapper.emplaceAndPop([self.datum])

        self.matchDetectionAndPose(lstDets, self.datum.poseKeypoints)

        # Build Message & Publish
        msgDetectionAndDirection            = DetectionAndDirection()
        msgDetectionAndDirection.header     = msg.header
        msgDetectionAndDirection.detections = self.detectionDirections

        self.publisherDetectionDirections.publish(msgDetectionAndDirection)

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

    
    def matchDetectionAndPose(self, detections, poses):
        '''
        Matches the Openpose detections with bounding boxes
        '''
        self.detectionDirections = []

        if poses.size > 1:

            for pose in poses:
                '''
                Check a few key positions (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose_25.png)
                0: Mid head
                1: Mid Torso
                2, 3, 4: Right Shoulder, Elbow, Arm
                5, 6, 7: Left Shoulder, Elbow, Arm
                '''

                # Check torso, right/left shoulder
                torso     = pose[1]
                rshoulder = pose[2]
                lshoulder = pose[5]
                
                for bbox in detections:
                    
                    if( self.withinBB(bbox, torso[0], torso[1]) or
                        self.withinBB(bbox, rshoulder[0], rshoulder[1]) or
                        self.withinBB(bbox, lshoulder[0], lshoulder[1])):
                    

                        if(rshoulder[0] > lshoulder[0]):
                            # Moving away from camera
                            cv2.circle(self.img, (torso[0], torso[1]), 9, (0, 255, 0), -1)
                            directionTowardsCamera = False
                        else:
                            # Moving towards camera
                            cv2.circle(self.img, (torso[0], torso[1]), 9, (0, 0, 255), -1)
                            directionTowardsCamera = True

                        # Build Messages
                        msgBoundingBox       = BoundingBox()
                        msgBoundingBox.Class = 'person'
                        msgBoundingBox.xmin  = bbox[0]
                        msgBoundingBox.ymin  = bbox[1]
                        msgBoundingBox.xmax  = bbox[2]
                        msgBoundingBox.ymax  = bbox[3]

                        msgBBoxDirection                        = BoundingBoxDirection()
                        msgBBoxDirection.boundingbox            = msgBoundingBox
                        msgBBoxDirection.directionTowardsCamera = directionTowardsCamera

                        self.detectionDirections.append(msgBBoxDirection)

                        # Once matched, move onto next pose
                        break 

                
    def withinBB(self, bbox, x, y):

        if( x >= bbox[0] and x <= bbox[2] and
            y >= bbox[1] and y <= bbox[3] ):
            return True
        else:
            return False


def main(args):
    yn = yact_node(debug = 1)
    rospy.init_node('yact_node', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down YACT node')
    
if __name__ == '__main__':
    main(sys.argv)