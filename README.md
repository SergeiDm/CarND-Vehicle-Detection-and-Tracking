# CarND-Vehicle-Detection-and-Tracking
## Project Description
The goal of the project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. The project includes the following steps:
- Data Exploration.
- Color Spaces Exploration.
- Techniques for features exctraction.
- Training classifier.
- Sliding Windows Search.
- Car Detection.

## Project files
The project includes the following folder/files:
- test_images - the folder with image examples for testing pipeline.
- output_images - the folder with results of testing all steps of pipeline.
- Vehicle_Detection_and_Tracking_Solution.ipynb - the script with pipeline.
- project_video.mp4 - the video to test pipeline.
- output_project_video.mp4 - the result of pipeline work on 'project_video.mp4'.

## Data Exploration
This section ('1. Data Exploration' section of 'Vehicle_Detection_and_Tracking_Solution.ipynb') describes dataset used for training classifier to distinguish car images from non-car images. These images were taken from this sources:
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
The Dataset includes 8792 car and 8968 non-car pictures. A picture is 64 x 64 pixels in 'png' file format. Dataset image examples:
![Dataset_image_examples](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/dataset_image_examples.jpg)

## Pipeline (single images)
