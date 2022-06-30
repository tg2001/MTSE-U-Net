### Visualization

This folder contains files for helping in visualization of 3D volumes and MRIs

For visualization purpose, there are three function in the visualization folder:

- The visualize_2d() will visualize any 2D image, for which, the 2D array should be passed as a parameter.
- The visualize_3d() will visualize any 3D volume, for which no parameters are needed, rather an input will be prompted after calling the function, where the path to the folder containing the volume(s) should be given. The volumes should be in NIfTI format (with extensions .nii or .nii.gz). This function will only work in ipython notebook (like colab), since it uses ipywidgets library.
- The brain_part_focus() will help in visualising either the brain as a whole, or any one of the brain component at a time.
