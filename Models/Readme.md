### Models

This folder contains the files for the code of the models we have developed and tested in our work (and included in our paper as ablation studies).

One file is dedicated to each model. Each file contains one function, named create_model(), which when called (no parameters need to be passed), will return the respective model.

However, the model will be initialised with random weights, so to use the model for the prediction purpose, the correct model weights should be loaded into the model.

We have included the model weights only for the MTSE U-Net model. For other models, you need to train them yourselves from scratch, after importing the model using the function.

Another alternative can be to directly load the model from the model files, in which case, these files are not needed. Again, this is only available for the MTSE U-Net model.

File named 'layer_6_mod_skip.py' is the code for MTSE U-Net model.

Follow the demo colab file for the tutorial on how to do it.
