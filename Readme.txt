
Software Requirements:
FLASHDeconv version: OpenMS-3.0.0
Unidec version: v.6.0.4
TopFD version: v.1.6.2
TopPIC version: v.1.6.2


Environment:
Python ==3.8.16
R == 4.3.1


For denoise_mode_related folder:

1. preprocess.py: This file contains the code for preprocessing the MS1 and MS2 data, including functions for data labeling, normalization, and other preprocessing steps.
2. train.py and train_ms2_model.py: These files contain the code for training the MS1 and MS2 models, respectively, and validating the models.
3. rawFile.py: This file is used to process the mzML files for denoising.



For FW_DIA_signal_process folder:

1. Deconvolution.py: This file contains the code for feature detection with MS1 data.
2. MS2Later.py: This file contains the code for extracting MS2 signals, calculating the elution profile similarity with MS1 data, and generating msalign files.
3. findcheck.py: This file contains the code for filtering the identified results.



Usage Guide:

1. Convert the .raw files into profile mzML files using MsConvert.
2. Denoise the profile mzML files using the functions in rawFile.py.
3. Convert the denoised mzML files into centroided mzML files using MsConvert.
4. For denoised MS1 data, use TopFD for deconvolution and store the results in the ./topfd/MS1 folder.
5. Perform MS1 feature detection using the functions in Deconvolution.py.
6. For denoised MS2 data, use the functions in M2Later.py to extract MS2 signals, match with precursor ions, and generate msalign files.
7. Send the msalign files from the previous step to TopPIC for identification.
8. Use the functions in findcheck.py to filter the identification results.

