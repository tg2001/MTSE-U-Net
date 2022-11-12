# MTSE U-Net

This is the repository for the following paper: 
```
MTSE U-Net: An architecture for segmentation and prediction of fetal brain and gestational age from MRI of brain
```

This paper is for performing three tasks:
- Segmenting the fetal brain into its seven major components: Intracranial space and extra-axial CSF spaces, Gray matter, White matter, Ventricles, Cerebellum, Deep gray matter and the Brainstem and spinal cord.
- Predicting the type of the fetal brain from the image ('Pathological' or 'Neurotypical').
- Predicting the gestational age of the fetus from the image.

We have used the FeTA 2.1 (Fetal Tissue Annotation) Dataset for the purpose of training and testing our models. The dataset can be downloaded from this website: [link](https://zenodo.org/record/4541606#.Yqb6HHVBw_A); after agreeing to the terms and conditions.

The repository contains the following folders:
- Dataset: Contains code for creating the dataset
- Testing: Contains code for using the model for performing segmentation and also for testing the performance of the models
- Visualisation: Contains code for visualising the dataset
- Models: Contains the code for all the models we tested out

(Information the codes and how to use them are uploaded in the readme file of their respective folders)

We have used Google Colaboratory for all the tasks.

The input to our model is a 3D array, containing all the 2D images that should be given to the model. Each image is of the shape (256x256).

The model files and saved weights are uploaded in this [google drive link](https://drive.google.com/drive/folders/1APnGfspLJN9vU_PL0TSXS52TjynUAvnO?usp=sharing).

The model will output an unsymmetrical 3D array:

- The first element is itself another array and the segmentation output having the shape (256x256x8) for each image (thus returning a 4D array for the segmentation output of all the images of the input), giving the probability of a pixel to be in each of the 8 classes: the seven brain components and the background; which is post-processed to return it in the required form. The final output will be a 3D array, having one segmented output image of shape (256x256) corresponding to each input image.
- The second output is the type prediction of the fetal brain. Three values will be returned corresponding to the following types: 'Age cannot be predicted from the slice', 'Pathological' or 'Neurotypical'. The one with the highest value will be chosen as the output.
- The third output is the age prediction of the fetus. This value is normalised, which will be changed back to the original form.

We have also created a demo colab file on how to use these modules for creating a dataset, visualising it, training and testing the models and getting outputs from them. It is uploaded in this [link](https://colab.research.google.com/drive/1Pn-iytS_8yZU3JegqfJcD4QnyF47UY3R?usp=sharing).

Libraries used in this this repository:
- Numpy
- Pandas
- Opencv
- OS
- Scikit-Learn
- Tensorflow 2
- Nibabel
- Time
- Tqdm
- Matplotlib
- Ipywidgets (required for visualization of 3D volumes; can only be used in Ipython notebooks)

Please cite our paper:
```
Gangopadhyay, T., Halder, S., Dasgupta, P. et al. MTSE U-Net: an architecture for segmentation, and prediction of fetal brain and gestational age from MRI of brain. Netw Model Anal Health Inform Bioinforma 11, 50 (2022). https://doi.org/10.1007/s13721-022-00394-y
```
