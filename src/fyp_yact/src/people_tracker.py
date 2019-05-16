#!/usr/bin/env python

import sys, os

import multiprocessing
import numpy as np

import cv2
import dlib

import rospy
from   sensor_msgs.msg import CompressedImage
import cv_bridge as bridge

from fyp_yact.msg import BoundingBox, BoundingBoxes

class yact_node:

    def __init__(self, debug = 0):
        pass



def main(args):
    pass

if __name__ == '__main__':
    main(sys.argv)