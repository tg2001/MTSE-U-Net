### Dataset

This folder contains the files for the preprocessing steps and the dataset creation step for training and testing the models.

For creating the dataset, the function create_dataset() in this folder should be called with the following parameters:
- First one is the path to the folder containing the 3D volumes (in the NIfTI format), which will form the input to the model.
- Second one is the path to the folder containing the 3D volumes (in the NIfTI format), which will form the corresponding segmented output.
  
  (The 3D volumes will be used to create 2D dataset for the model)
- The third parameter is the path to the file containing the fetal brain type and age information for each 3D volume.
- The fourth parameter 'n' will determine the number of 2D images taken from each axis of each volume. Along each axis, the 3D volume is divided into 'n' parts and 1 slice from each part is taken to create the dataset.
- The last parameter 's' will determine the test set size (in numbers or fraction), keeping the remaining for the train set. s=0 means no splitting will take place.

This function will return four things if 's' is not equal to 0: train and test split for the input, followed by train and test split for the output.

Otherwise, the input and output dataset will be returned as a whole.

This function is written mainly to work on the FeTA Dataset, however, it can be easily be modified to work on any other dataset by making some minor changes:
- Commenting/deleting out 96th line and the code portions where the 'non_blur' variable is used. This is exclusively for feta dataset.
- Updating 106th line to the required naming pattern for getting the corresponding segmented output from the output folder.
- In case of the feta dataset, a single .tsv file had both type and age prediction information. However, this might not be the case for some other dataset. Thus, this part can also be edited to the required form (for example, two variables for accessing data from two different files).
