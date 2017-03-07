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

![3D color space repr](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/3D_color_space_repr.jpg)

HLS and HSV 3D color space representations make car colors more distinguishable because of saturation. It may be used in features exctraction.

### Spatial Binning of Color
Feature vector consisting on raw pixels can be useful for car detection, but instead of using of three color channels of a full resolution image, I decreased resolution to 8 x 8 pixels and applied 'numpy.ravel' function to create a flattened array. Before this, HLS color space conversion was used (function 'bin_spatial' in '2.2. Spatial Binning of Color' section). As stated above, this color space can make car colors more distinguishable. 

Example:

![Spatial binning of color](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Spatial_binning_of_color.jpg)

From pictures above, we can see that all three HLS channels can be used for creating feature vector. Other color spaces also may be used.

### Histograms of Color
Another tool for creation feauture vector is histograms of raw pixel intensity. Concatenating histograms of different color channels we create a pattern of particular number of bins and pixels intensity range. In this project I used HSV color space for building of color channels histograms (function 'color_hist' in '2.3. Histograms of Color'). HSV color space gave a good distinguishable representation of car and non-car images due to in particular its saturation channel. 

Here are examples (bins=12):

![Histograms of color](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Histograms_of_color.jpg)

All HSV channels differentiate car and non-car images. From histograms we can see that an image with monotonous colors (2 pictures in the last two rows) gives bigger values for histograms than images which include different colors.

### Histogram of Oriented Gradients (HOG)
This features descriptor in distinction from previous two based on edge detection (of course, it is derivative of color, but not a color itself). HOG can be calculated with 'hog' function from scikit-image library. Deriving hog features is performed in 'get_hog_features' function (section '2.4. Histogram of Oriented Gradients (HOG)').

Here are output examples (orient = 9, pix_per_cell = 8, cell_per_block = 2):

![Histogram of oriented gradients](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Histogram_of_oriented_gradients.jpg)

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
- 51.73 Seconds to extract features...
- Feature vector length: 372
- 6.31 Seconds to train SVC...
- Test Accuracy of SVC =  0.9713
- SVC predicts:  [ 1.  1.  1.  0.  0.  0.  0.  0.  1.  0.] For these 10 labels:  [ 1.  1.  1.  0.  0.  0.  0.  0.  1.  0.]
- 0.01563 Seconds to predict 10 labels with SVC"

### Sliding Windows Search
With sliding window implementation we extract areas where trained classifier decides given area is a car or not. For sliding window search we have to choose window size, region in an image where window will slide and overlapping parameter. A good idea is to use different window size in different regions of an image (multi-scale windows). Than more windows are used than slower pipeline.

I applied three window sizes in different regions. My choice is based on:
- Different sizes of window allowed to match their size with a car size (which is varied in different distances (position in an image)). Window with bigger sizes are used at the bottom of an image, while the smaller are at the top.
- This creates better covering region of interest on an image with reasonable computational cost.

Finally, I used the following parameters for sliding windows search:
- three window sizes: 1) 64 x 64, 2) 128 x 128, 3) 192 x 192 pixels.
- three regions for searching: x coordinate is the same - (0, 1280), y coordinate: 1) (390, 500), 2) (390, 590), 3) (390, 670).
- three overlapping parameters: 1) (0.5, 0.5), 2) (0.6, 0.5), 3) (0.5, 0.5).

Here are results:

![Sliding windows](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Sliding_windows.jpg)

Sliding window implementation is in '2.6. Sliding Windows Search' section, function 'slide_window'.

Sliding window technique may give duplicates and false positives. Examples:

![Duplicates and false positives](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Duplicates_and_false_positives.jpg)

To exclude duplicates and false positives, a heat-map was created in which overlapping detections combined (for every pixels inside box 1 is added, see 'add_heat' function in '2.6. Sliding Windows Search' section), so duplicates are combined. Applying threshold for value of pixels we exclude false positives.

For calculating how many cars in a heatmap, I used 'label' function from scipy.ndimage.measurements.

Finally all stages can be seen here:

![Pipeline images](https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_images/Pipeline_images.jpg)

The results, in most cases, coincide with number of cars.

## Pipeline(video)
Pipeline for video include the following steps:
- Sliding windows search with trained classifier.
- Car detections with applying heatmap for excluding duplicates and false positives.

Parameters for sliding windows, threshold for heatmap are the same as for processing images.

Moreover, I combined functions for lane lines (used in P4: Advanced-Lane-Finding) and vehicle detections in order to show their mutual detections.

Here's a link to my video result: https://github.com/SergeiDm/CarND-Vehicle-Detection-and-Tracking/blob/master/output_project_video.mp4.

## Discussion
Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail. 
he pipeline which was applied here has the folowing main steps:

    Camera Calibration.
    Distortion Correction.
    Creating binary images.
    Perspective transform.
    Detection Lane Lines pixels. Steps 'Creating binary images' and 'Detection Lane Lines pixels' have a lot of hyperparameters (e.g. min, max threshold, color space transformation and so on) and assumptions (we have both right and left lane lines on image, conditions allow us to identify lines). When we tune hyperparameters, we try to fit the pipeline for certain given exmaples. It may lead us to overfitting (in terms of neural network). It means when we meet video for example with a new weather conditions we may not to clearly identify lines.

The cases where the algorithm given here is likely to fail may be the following:

    sunny wheather conditions,
    shadows from objects,
    changing road surface color,
    a car going ahead,
    double lines,
    elements like barriers with yellow or white coloring,
    poor images.

In cases listed above, lane lines pixels are complicated to distinguish from surrounding area. So for improving algorithm the following steps seem reasonable:

    collect cases with different conditions (not only weather conditions, but also shadows, surface color and so on).
    define a recovery algorithm when we lost lane lines track.
    define some metrics to check that the results are correct and correction plan if they are not.
    use other criteria for defining a car position.






