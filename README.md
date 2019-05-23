# YACT: Yet Another Crowd Tracker

This repository contains the ROS nodes for tracking and calculating the trajectories of humans in a crowd, for use in my Final Year Project.

## Node: yact_node

The node depends on the detections of people/person objects from the yolo detector.

## Submodules:

* ~~[SORT](https://github.com/alaksana96/sort/tree/master)~~
    * ~~Based off __abewley__'s implementation of the SORT algorithm~~

* [Deep SORT](https://github.com/alaksana96/deep_sort/tree/master)
    * Deep SORT was a major improvement to the basic SORT algorithm.
    * Better tracking of objects due to feature generation of Bounding Boxes as well as using a learned cosine distance as a metric to compare tracked objects to new detections.
    * Based off [nwojke](https://github.com/nwojke)'s implementation.
        * Modified to run on ``tensorflow-gpu==1.4.0`` (CUDA 8.0 and cuDNN6 on Ubuntu 16.04).

    * [Youtube Video](https://youtu.be/1Br1ZKIr9FY)

## Custom Messages

Open the ``src/fyp_yact/msg`` folder to see the custom messages:

* **`BoundingBox`** 

    Contains the individual detections.

    ```
    string Class
    float64 probability
    int64 xmin
    int64 ymin
    int64 xmax
    int64 ymax
    ```

* **`CompressedImageAndBoundingBoxes`** 

    Contains the list of detections.

    ```
    Header header
    string format
    uint8[] data
    BoundingBox[] bounding_boxes
    ```
## Topics

### Subscribed Topics

* **`/yolo_detector/output/compresseddetections`** ([sensor_msg::CompressedImageAndBoundingBoxes])

    Subscribe to the compressed image topic from the Hololens in the JPEG format and the detections from YOLO.