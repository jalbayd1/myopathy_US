This repo contains ultrasound images, ultrasound image segmentations, and patient diagnosis charts for data used in the paper:

(Under Review)

The repo containes the following files:

* `AlexNet_Myopathy`: the CNN architecture in Keras, which you can pass US images through to obtain classifications
* `im_muscle_chart.mat`: a Matlab table the muscle type, diagnosis, and deidentified patient ID of each US image
* `patient_chart.mat`: a Matlab table listing the diagnosis of each deidentified patient ID (this information is also contained in `im_muscle_chart.mat` but this table gives a patient-centric overview as opposed to listing each image that was acquired)

You can find the data files in our [latest release](https://github.com/jalbayd1/myopathy_US/releases/latest), which contains:
* `im2D.mat`: Matlab cell arrays containing the US images. The images are ordered according to the row numbers of the table `im_muscle_chart.mat`.
* `im2DSeg.mat`: Matlab cell array containing the US image segmentations denoting muscle and fat image regions.
* `weights_PXX_FOLD_1.hdf5`: Keras model weights which must be loaded in AlexNet_Myopathy.py to do classifications for a given experiment.