import numpy as np
import os
import nibabel as nib

from ipywidgets import interact, interactive, fixed, AppLayout
import matplotlib.pyplot as plt
# %matplotlib inline





# Function to visualise a 2D slice by resizing the values in the image to the range 0-255, if required after checking the conditions
# If the max element is 0, then there's no need of resizing
# If the maximum element is less than 255 and is an integer, then also, resizing is not required

def visualize_2d(img):

  m = max(np.reshape(img, (-1)))
  l = [np.uint8, int, np.int8, np.int16]

  if m>0 and not (m<=255 and (type(m) in l)):
    img = np.array(img*255/m, dtype=np.uint8)
  else:
    img = np.array(img, dtype=np.uint8)

  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.show()
  
  
  
  
  
# Function to visualise the 3D MRI along each axis

def explore_image(layer, n, data):

  if n==0:
    visualize_2d(data[layer, :, :])     # Moving parallel to saggital plane

  elif n==1:
    visualize_2d(data[:, layer, :])     # Moving parallel to coronal plane

  elif n==2:
    visualize_2d(data[:, :, layer])     # Moving in horizontal plane

  return layer
  
  
  
  
  
  
# Creating a slider to select the layer to visualise in each axis followed by visualising the image

def return_3d(select_file, c=0, x=0):

  if c==1:    # c = 1 implies first loading the 3D MRI from the file location, and then visualising
    if not x==0:
      data1 = nib.load(x+select_file).get_fdata()

    else:
      data1 = nib.load(select_file).get_fdata()

  else:     # c = 0 implies that the file is already loaded for visualising
    data1 = select_file
    
  m = max(data1.reshape(-1))
  data = np.array((data1*255/m), dtype=np.uint8)    # Clipping the values from 0 to 255

  # Creating 3 interactive sliders for each of the 3 axes

  i1 = interactive(explore_image, layer=(0, data.shape[0]-1), n = fixed(0),data = fixed(data))
  i2 = interactive(explore_image, layer=(0, data.shape[1]-1), n = fixed(1),data = fixed(data))
  i3 = interactive(explore_image, layer=(0, data.shape[2]-1), n = fixed(2),data = fixed(data))

  # Layout to visualise all the three axes side by side
  layout = AppLayout(header=None, left_sidebar=i1, center=i2, right_sidebar=i3, footer=None, pane_widths=[1, 1, 1])
  display(layout)
  
  
  
  
  
# Function to create an interface for visualising the 3D MRI along the 3 axes

def visualize_3d():
  x = input('Enter path containing image folder: ')   # Taking path of the folder containing the 3D files

  if not x[-1]=='/':
    x = x+'/'
  l = os.listdir(x)   # Getting the list of files from the directory given by user 

  interact(return_3d, select_file=l, c=fixed(1), x=fixed(x))   # Creating a drop-down menu for selecting the 3D image to visualise
  
  
  


# This function will help in visualising and focusing at any particular brain part of all the segmented parts 
# and the whole brain after removing the noises

def brain_part_focus(data1, data2):

  print('Enter the segmented brain part to veiw:\n')
  print('1. Intracranial space and extra-axial CSF spaces')
  print('2. Gray matter')
  print('3. White matter')
  print('4. Ventricles')
  print('5. Cerebellum')
  print('6. Deep gray matter')
  print('7. Brainstem and spinal cord')
  print('8. Segmented brain without noise')

  while True:
    i = int(input('\nEnter your choice: '))

    if i<1 or i>8:
      print('Invalid choice. Retry!')

    else:
      break

  if i==8:
    d1 = np.where(data2>0, data1, 0)    # Keeping everything which correspond to any brain part

  else:
    d1 = np.where(data2==i, data1, 0)   # Keeping only the pixels which correspond to the brain part asked for visualisation

  visualize_2d(d1)




