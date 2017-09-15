This repo contains ultrasound images, ultrasound image segmentations, and patient diagnosis charts for data used in the paper:

*Automated diagnosis of myositis from muscle ultrasound: Exploring the use of machine learning and deep learning methods(https://doi.org/10.1371/journal.pone.0184059)*

Installation is as simple as running `pip install -r requirements.txt`

The repo contains `AlexNet_Myopathy`: the CNN architecture in Keras, which you can pass US images through to obtain classifications. An example is in the main() function.

You can find the data and model weights in our [latest release](https://github.com/jalbayd1/myopathy_US/releases/latest), which contains:
* `PatientData.mat`: The main data file, which contains:
    * `im`: Matlab cell arrays containing the US images. The images are ordered according to the row numbers of the table `im_muscle_chart`.
    * `imSeg.mat`: Matlab cell array containing the US image segmentations denoting muscle and fat image regions.
    * `im_muscle_chart`: a Matlab table of the muscle type, diagnosis, and deidentified patient ID of each US image.
    * `patient_chart`: a Matlab table listing the diagnosis of each deidentified patient ID (this information is also contained in `im_muscle_chart` but this table gives a patient-centric overview as opposed to listing each image that was acquired).
* `PatientImages_PLOS2017.xlsx`: Similar to image_muscle_chart described above, but in an Excel spreadsheet format.
* `weights_ProblemA_FOLD_1.hdf5`: Keras model weights which must be loaded in AlexNet_Myopathy.py to do classifications for Problem A.
* `weights_ProblemB_FOLD_1.hdf5`: Keras model weights which must be loaded in AlexNet_Myopathy.py to do classifications for Problem B.
* `weights_ProblemC_FOLD_1.hdf5`: Keras model weights which must be loaded in AlexNet_Myopathy.py to do classifications for Problem C.
