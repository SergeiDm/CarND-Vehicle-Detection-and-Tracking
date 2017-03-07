# CarND-Vehicle-Detection-and-Tracking
## Project Description
The goal of the project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. The project includes the following steps:
- Data Exploration.
- Color Spaces Exploration.
- Techniques for features exctraction.
- Training classifier.
- Sliding Windows Search.
- Car Detection.
Vehicle detection in this project is built on considering different parts of an image and classificating them as a car or not car due to their feautures.

## Project files
The project includes the following folder/files:
- test_images - the folder with image examples for testing pipeline.
- output_images - the folder with results of testing all steps of pipeline.
- Vehicle_Detection_and_Tracking_Solution.ipynb - the script with pipeline.
- project_video.mp4 - the video to test pipeline.
- output_project_video.mp4 - the result of pipeline work on 'project_video.mp4'.

## Data Exploration
This section (see '1. Data Exploration' section of 'Vehicle_Detection_and_Tracking_Solution.ipynb') describes dataset used for training classifier to distinguish car images from non-car images. These images were taken from this sources:
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip.
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip.

The Dataset includes 8792 car and 8968 non-car pictures. A picture is 64 x 64 pixels in 'png' file format. Dataset image examples:
![Dataset_image_examples](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/dataset_image_examples.jpg)

## Pipeline (single images)
### Color Spaces Exploration
Color spaces exploration is a way to find what kind of feature (color channel) can be used to distinguish a target object (in our case - a car) from other objects and background.
In this project, function 'colorspace_repr' (section '2.1. Color Spaces Exploration') was defined to represent an image in 3D color space, e.g. HLS and HSV (commonly used in image processing). 

Example:
![3D_color_space_repr](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/3D_color_space_repr.jpg)

HLS and HSV 3D color space representations make car colors more distinguishable because of saturation. It may be used in features exctraction.

### Spatial Binning of Color
Feature vector consisting on raw pixels can be useful for car detection, but instead of using of three color channels of a full resolution image, I decreased resolution to 8 x 8 pixels and applied 'numpy.ravel' function to create a flattened array. Before this, HLS color space conversion was used (see function 'bin_spatial' in '2.2. Spatial Binning of Color'). As stated above, this color space can make car colors more distinguishable. 

Example:
![Spatial_binning_of_color](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Spatial_binning_of_color.jpg)

From pictures above, we can see that all three HLS channels can be used for creating feature vector. Other color spaces also may be used.

### Histograms of Color







