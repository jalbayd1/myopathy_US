This repo contains ultrasound images, ultrasound image segmentations, and patient diagnosis charts for data used in the paper:

(Under Review)

The repo containes the following files:

* `im_muscle_chart.mat`: a Matlab table the muscle type, diagnosis, and deidentified patient ID of each US image
* `patient_chart.mat`: a Matlab table listing the diagnosis of each deidentified patient ID (this information is also contained in `im_muscle_chart.mat` but this table gives a patient-centric overview as opposed to listing each image that was acquired)
* `im2D_a.mat` - `im2D_f.mat`: Matlab cell arrays containing the US images. The images are ordered according to the row numbers of the table `im_muscle_chart.mat`.  Since GitHub has a file size limit of 100MB, the 2D image cell array has been split into multiple Matlab files.  These cell arrays should be concatenated into one cell array before cross-referencing with `im_muscle_chart.mat`.
* `im2DSeg.mat`: Matlab cell array containing the US image segmentations denoting muscle and fat image regions.