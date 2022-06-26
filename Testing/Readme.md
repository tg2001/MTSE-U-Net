### Testing

This folder contains files for outputting, testing and evaluating the models.

For testing or getting output from the model, first the model should be loaded, then pred_and_eval() in this folder should be called with the following parameters:
- The model to be used should be passed as the first input.
- The input(s) to the model should be passed next.
  
For getting output from the model, the above two inputs are enough. In this case, the model will return an asymmetric 3D array containing the following: the first array is itself another symmetric 3D array, containing the segmented output to all the 2D input images, the next two arrays are each 1D arrays, the first one containing the predicted type of all the input images and the second one containing their predicted gestational ages in weeks.
  
For evaluating the model's performance, the next parameters should be passed along with the above ones:
- The actual output(s), for the function to compare with.
- The last parameter will decide whether to return the average value of the segmentation task evaluation metrics for all the classes or to return detailed evaluation with respect to each class for all the images in the test set. This parameter is for the segmentation task only.

Evaluation metrics used: 
- For segmentation task: Precision, Sensitivity, Jaccard Similarity, Dice Score, Accuracy 
- For type prediction task: Accuracy
- For age prediction task: Mean Absolute Error

Accuracy (for the segmentation task) won't be returned when the detailed evaluation for all the classes are returned.
