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

Example (spatial size = (8,8)):

![3D_color_space_repr](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/3D_color_space_repr.jpg)

HLS and HSV 3D color space representations make car colors more distinguishable because of saturation. It may be used in features exctraction.

### Spatial Binning of Color
Feature vector consisting on raw pixels can be useful for car detection, but instead of using of three color channels of a full resolution image, I decreased resolution to 8 x 8 pixels and applied 'numpy.ravel' function to create a flattened array. Before this, HLS color space conversion was used (function 'bin_spatial' in '2.2. Spatial Binning of Color' section). As stated above, this color space can make car colors more distinguishable. 

Example:

![Spatial_binning_of_color](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Spatial_binning_of_color.jpg)

From pictures above, we can see that all three HLS channels can be used for creating feature vector. Other color spaces also may be used.

### Histograms of Color
Another tool for creation feauture vector is histograms of raw pixel intensity. Concatenating histograms of different color channels we create a pattern of particular number of bins and pixels intensity range. In this project I used HSV color space for building of color channels histograms (function 'color_hist' in '2.3. Histograms of Color'). HSV color space gave a good distinguishable representation of car and non-car images due to in particular its saturation channel. 

Here are examples (bins=12):

![Histograms_of_color](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Histograms_of_color.jpg)

All HSV channels differentiate car and non-car images. From histograms we can see that an image with monotonous colors (2 pictures in the last two rows) gives bigger values for histograms than images which include different colors.

### Histogram of Oriented Gradients (HOG)
This features descriptor in distinction from previous two based on edge detection (of course, it is derivative of color, but not a color itself). HOG can be calculated with 'hog' function from scikit-image library. Deriving hog features is performed in 'get_hog_features' function (section '2.4. Histogram of Oriented Gradients (HOG)').

Here are output examples (orient = 9, pix_per_cell = 8, cell_per_block = 2):

![Histogram_of_oriented_gradients](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Histogram_of_oriented_gradients.jpg)

In pictures above represented channels of color spaces give 'readable' information, but finally I used GRAY color space.

### Trainig classifier
In '2.5. Training classifier' section I created 'single_img_features' function which exctract image features by using:
- Spatial Binning of Color. Parameters: color space - HLS, all channels, spatial size - 8 x 8 pixels.
- Histograms of Color. Parameters: color space - HSV, all channels, number of bins - 12.
- Histogram of Oriented Gradients (HOG). Parameters: color space - GRAY, orientations - 9, size (in pixels) of a cell - 18, Number of cells in each block - 2.

Ideas for feature extractors and choosing values for parameters were the following:
- Using different features extractors may help to provide stable results in different conditions: size, shape, color, postion of a car.
- Using HLS and HSV color spaces as ones which give a good car color representation.
- Using values of parameters which on the one hand outputs acceptable results, but on the other hand provide keep computational cost reasonable small. In this case, once again, HLS and HSV channel are good choice.

For classification I used linear support vector classifier, which is a good choice for speed and accuracy.

I also tried different parameters and classifiers, but results were worse or computational cost was higher. Here are results of training classifier:
"51.73 Seconds to extract features...
Feature vector length: 372
6.31 Seconds to train SVC...
Test Accuracy of SVC =  0.9713
SVC predicts:  [ 1.  1.  1.  0.  0.  0.  0.  0.  1.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  0.  0.  0.  0.  1.  0.]
0.01563 Seconds to predict 10 labels with SVC"

### Sliding Windows Search



## Pipeline(video)



## Discussion







