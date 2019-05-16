# YACT: Yet Another Crowd Tracker

This repository contains the ROS nodes for tracking and calculating the trajectories of humans in a crowd, for use in my Final Year Project.

## Node: yact_node

The node depends on the detections of people/person objects from the yolo detector.

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

* **`BoundingBoxes`** 

    Contains the list of detections.

    ```
    Header header
    Header image_header
    BoundingBox[] bounding_boxes
    ```
## Topics

### Subscribed Topics

* **`/yolo_detector/output/compressed`** ([sensor_msg::CompressedImage])

    Subscribe to the compressed image topic from the Hololens in the JPEG format.

* **`/yolo_detector/output/detections`** ([fyp_yact::BoundingBoxes])

    Subscribe to the detections from the YOLO detector.