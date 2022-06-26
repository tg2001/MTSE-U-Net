import numpy as np
import cv2 as cv





''' 
Function to remove all the black slices (i.e. removing the 2D slices completely filled with 0) along any particular axis

This function is used as a preprocessing step to enable the collection of layers (for training) 
with have atleast a small part of the brain MRI (in any axis) for the model to learn segmenting it

We need to apply this step to both input and its segmented output image as it will be used for training. 
Thus data1 and data2 below correspond to the input and the output.

'n' parameter determines the axis along which the black slices will be removed 
'''

def reduce_2d(data1, data2, n):

  i = 0
  c = 0

  if n==2:
    while True:
      x = np.reshape(data1[:, :, i], (-1))    # i determines the current slice under consideration. It can be 0 (first slice) or -1 (last slice)

      # Getting the maximum and minimum values of the current slice in the 2nd axis
      m1 = max(x)
      m2 = min(x)

      if m1==m2:    # If the two values are equal, then that layer is completely filled with 0, thus eliminated
        if c==0:    # c=0 indicates eliminating the black slices from front (i.e. from 0 along this axis)
          data1 = data1[:, :, 1:]
          data2 = data2[:, :, 1:]

        else:       # c=1 indicates eliminating the black slices from end (i.e. from 255 along this axis)
          data1 = data1[:, :, :-1]
          data2 = data2[:, :, :-1]

      elif c==0:    # c will change to 1 when there will be no more black slices from the front
        i = -1
        c = 1

      else:   # The control will break out of the loop when no more black slices from the front or end will be there
        break

  elif n==1:
    while True:
      x = np.reshape(data1[:, i, :], (-1))    # i determines the current slice under consideration. It can be 0 (first slice) or -1 (last slice)

      # Getting the maximum and minimum values of the current slice in the 1st axis
      m1 = max(x)
      m2 = min(x)

      if m1==m2:      # If the two values are equal, then that layer is completely filled with 0, thus eliminated
        if c==0:      # c=0 indicates eliminating the black slices from front (i.e. from 0 along this axis)
          data1 = data1[:, 1:, :]
          data2 = data2[:, 1:, :]

        else:         # c=1 indicates eliminating the black slices from end (i.e. from 255 along this axis)
          data1 = data1[:, :-1, :]
          data2 = data2[:, :-1, :]

      elif c==0:      # c will change to 1 when there will be no more black slices from the front
        i = -1
        c = 1

      else:   # The control will break out of the loop when no more black slices from the front or end will be there
        break

  else:
    while True:
      x = np.reshape(data1[i, :, :], (-1))    # i determines the current slice under consideration. It can be 0 (first slice) or -1 (last slice)

      # Getting the maximum and minimum values of the current slice in the 0th axis
      m1 = max(x)
      m2 = min(x)

      if m1==m2:      # If the two values are equal, then that layer is completely filled with 0, thus eliminated
        if c==0:      # c=0 indicates eliminating the black slices from front (i.e. from 0 along this axis)
          data1 = data1[1:, :, :]
          data2 = data2[1:, :, :]

        else:         # c=1 indicates eliminating the black slices from end (i.e. from 255 along this axis)
          data1 = data1[:-1, :, :]
          data2 = data2[:-1, :, :]

      elif c==0:      # c will change to 1 when there will be no more black slices from the front
        i = -1
        c = 1

      else:   # The control will break out of the loop when no more black slices from the front or end will be there
        break

  return data1, data2
  
  
  
  
  
  

  
# Flipping the images to introduce some translational independence

def flip(d1, d2, i):    # Possible values of i are (-1, 0, 1, 2)

  if i==2:
    return d1, d2   # No flipping
  else:
    return cv.flip(d1, i), cv.flip(d2, i)   # value of i determines the flip








# Blurring the images to consider mri which got blurred out or stretched due to fetal movement

def blur(x, i):   # Possible values of i are (0, 1, 2)

  if i==0:  # No Blurring
    return x

  else:     # Some blurring will be there depending on the value of f below
    f = np.random.randint(3)

    if f==0:
      return cv.GaussianBlur(x, (11, 11), 0)  # Normal Blurring

    elif f==1:
      return cv.GaussianBlur(x, (15, 1), 0)   # Horizontal stretching

    else:
      return cv.GaussianBlur(x, (1, 15), 0)  # Vertical stretching
      
      
      
