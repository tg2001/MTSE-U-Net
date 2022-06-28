import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur








# Function for splitting the dataset into train and test set and normalising the values

def split_dataset(input_mri, output_mri, s):

  input_mri = np.array(input_mri, dtype=np.uint8)
  output_mri = np.array(output_mri, dtype=np.uint8)
  mri_type = np.array(mri_type, dtype=np.uint8)
  age = np.array(age, dtype=np.float16)

  if s==0:
    # Normalising the input to the range 0 to 2
    input_mri = np.array(input_mri/128, dtype=np.float16)
    output_mri = (output_mri, mri_type, age)
    return input_mri, output_mri
  
  else:
    # Splitting up the dataset into training and testing set
    X_train, X_test, out1, out2, type1, type2, age1, age2 = train_test_split(input_mri, output_mri, mri_type, age, test_size=s, random_state=38)

    del input_mri, output_mri

    # Normalising the input to the range 0 to 2
    X_train = np.array(X_train/128, dtype=np.float16)
    X_test = np.array(X_test/128, dtype=np.float16)

    # Normalising the age to the range 0 to 2, where 0 indicating that the age (or type) of that particular slice can not be determined
    age1 = np.where(age1>0, (age1-15)/10, 0)
    age2 = np.where(age2>0, (age2-15)/10, 0)

    # Test set for multiple output
    y_train = (out1, type1, age1)
    y_test = (out2, type2, age2)

    del out1, out2
    return X_train, X_test, y_train, y_test









'''  
     Function to create the dataset for both training ang testing the model
     This will take 5 parameters: the first two are the paths to the input MRIs and their corresponding output segmentations,
     the third parameter will be the path to the file containing the fetal brain type and age information,
     the fourth parameter will determine the number 2D images taken from each axis of each MRI,
     and the last parameter will determine the test set size (in numbers or decimal), keeping the remaining for the train set size
     
     Larger n value means more images will be taken.
     s=0 means that no splitting will take place
     
     It will return four things if s not equals to 0: train and test split for the input, followed by train and test split for the output,
     otherwise, the input and output dataset will be returned as a whole 
'''


def create_dataset(path1, path2, f, n=40, s=0.05):

# Creating the dataset for both training ang testing the model

  # path1 is the folder containing the input for the model
  # path2 is the folder containing the segmented outputs
  # The 'f' file contains the information on the type and age of each MRI

  if not path1[-1]=='/':
    path1 = path1+'/'

  if not path2[-1]=='/':
    path2 = path2+'/'
    
  l = os.listdir(path1)     # storing the file names of the input MRI folder
  num = len(l)

  df = pd.read_csv(f, sep='\t')

  # non_blur list consists of the first 7 characters of those mri file's names, which are not suitable for blurring
  non_blur = ['sub-009', 'sub-005', 'sub-008', 'sub-007', 'sub-004', 'sub-002', 'sub-015', 'sub-023', 'sub-016', 'sub-017', 'sub-022', 'sub-021', 'sub-020', 'sub-062', 'sub-071', 'sub-078']

  input_mri = []
  output_mri = []
  mri_type = []
  age = []

  for i in tqdm(range(80), desc="Executing", ncols=75):

    f1 = l[i]                      # Getting one input mri file in each iteration
    f2 = f1[:-10]+'dseg'+f1[-7:]    # Getting the corresponding segmented output file
    
    # Example input mri file name and it corresponding segmented output file name respectively:
    # 'sub-001_rec-mial_T2w.nii.gz' and 'sub-001_rec-mial_dseg.nii.gz'

    # Loading the two files
    data1 = nib.load(path1+f1).get_fdata()

    data2 = nib.load(path2+f2).get_fdata()
    data2 = np.array(data2, dtype=np.uint8)   # Changing the type of each element from float64 to int8

    x = df[df.participant_id==f1[:7]]   # Getting the fetal brain type and age information of the corresponding input mri
    y1 = x['Pathology'].tolist()[0]

    if y1=='Pathological':    # Encoding string type to integers
      y1 = 1    # 'Pathological' type means 1
    else:
      y1 = 2    # 'Neurotypical' means 2

    # y2 = round(x['Gestational age'].tolist()[0])
    y2 = x['Gestational age'].tolist()[0]

    '''
    From the next line of code, the function 'reduce_2d' is applied to every axis
    in each mri

    Then for each axis (along which the function is applied), the resulting mri
    is divided into 'n' parts (decided by the variable n) and 1 slice from each
    part is taken to create the dataset

    After that the transformations are applied to the selected slices and added 
    to the dataset

    All of the above transformations (except for blurring) are applied to both
    input mri and its corresponding segmented output
    '''

    n = 40

    data10, data20 = reduce_2d(data1, data2, 0)   # Applying the function along 0th axis
    s0 = np.asarray(data10).shape[0]    # The resulting shape along that axis
    t0 = int(s0/n)    # Dividing the mri into n parts along the 0th axis
    r0 = list(range(0, s0, t0))   # Getting the index of slices along 0th axis to be considered from each part
    l0 = len(r0)

    data11, data21 = reduce_2d(data1, data2, 1)   # Applying the function along 1st axis
    s1 = np.asarray(data11).shape[1]    # The resulting shape along that axis
    t1 = int(s1/n)    # Dividing the mri into n parts along the 1st axis
    r1 = list(range(0, s1, t1))   # Getting the index of slices along 1st axis to be considered from each part
    l1 = len(r1)

    data12, data22 = reduce_2d(data1, data2, 2)   # Applying the function along 2nd axis
    s2 = np.asarray(data12).shape[2]    # The resulting shape along that axis
    t2 = int(s2/n)    # Dividing the mri into n parts along the 2nd axis
    r2 = list(range(0, s2, t2))   # Getting the index of slices along 2nd axis to be considered from each part
    l2 = len(r2)

    # Along each axis, the selected slices are taken, transformations are applied and added to the dataset
    c = 1
    for j in r2:
      # break

      d1 = data12[:, :, j]
      d2 = data22[:, :, j]

      f = np.random.randint(-1, 3)
      d1, d2 = flip(d1, d2, f)    # Applying flip transformation

      if f1[:7] not in non_blur:  # Applying blurring if the mri is not in the non-blur list
        f = np.random.randint(2)
        d1 = blur(d1, f)

      m = max(d1.reshape(-1))
      d1 = np.array(d1*255/m, dtype=np.uint8)   # Clipping the input element values to the range 0 to 255

      input_mri.append(d1)
      output_mri.append(d2)

      # Code to append the age and type of the corresponding mri slice if the slice is away from the corners of the mri
      # Otherwise append 0 to both the type and age to indicate that they can't be predicted from the corner slices of the mri

      if (c>(4*n/10)) ^ ((l2-c)<4*n/10):
        mri_type.append(y1)
        age.append(y2)
        c += 1
        # visualize_2d(d1)

      else:
        mri_type.append(0)
        age.append(0)
        c += 1

    c = 1
    for j in r1:
      # break

      d1 = data11[:, j, :]
      d2 = data21[:, j, :]

      f = np.random.randint(-1, 3)
      d1, d2 = flip(d1, d2, f)    # Applying flip transformation

      if f1[:7] not in non_blur:  # Applying blurring if the mri is not in the non-blur list
        f = np.random.randint(2) 
        d1 = blur(d1, f)

      m = max(d1.reshape(-1))
      d1 = np.array(d1*255/m, dtype=np.uint8)   # Clipping the input element values to the range 0 to 255

      input_mri.append(d1)
      output_mri.append(d2)

      # Code to append the age and type of the corresponding mri slice if the slice is away from the corners of the mri
      # Otherwise append 0 to both the type and age to indicate that they can't be predicted from the corner slices of the mri

      if (c>(4*n/10)) ^ ((l1-c)<4*n/10):
        mri_type.append(y1)
        age.append(y2)
        c += 1
        # visualize_2d(d1)

      else:
        mri_type.append(0)
        age.append(0)
        c += 1

    c = 1
    for j in r0:
      # break

      d1 = data10[j, :, :]
      d2 = data20[j, :, :]

      f = np.random.randint(-1, 3)
      d1, d2 = flip(d1, d2, f)    # Applying flip transformation

      if f1[:7] not in non_blur:  # Applying blurring if the mri is not in the non-blur list
        f = np.random.randint(2)
        d1 = blur(d1, f)

      m = max(d1.reshape(-1))
      d1 = np.array(d1*255/m, dtype=np.uint8)   # Clipping the input element values to the range 0 to 255

      input_mri.append(d1)
      output_mri.append(d2)

      # Code to append the age and type of the corresponding mri slice if the slice is away from the corners of the mri
      # Otherwise append 0 to both the type and age to indicate that they can't be predicted from the corner slices of the mri

      if (c>(4*n/10)) ^ ((l0-c)<4*n/10):
        mri_type.append(y1)
        age.append(y2)
        c += 1
        # visualize_2d(d1)

      else:
        mri_type.append(0)
        age.append(0)
        c += 1

  # Creating some a completely black slice (i.e. the one filled with 0) and appending it few times to the dataset for training the model 
  # on completely black images as well. The age and type in these cases will be 0, since they can't be determined from the black slices

  n_b = 30
  black = np.zeros((n_b, 256, 256), dtype=np.uint8)

  input_mri.extend(black)
  output_mri.extend(black)
  mri_type.extend([0]*n_b)
  age.extend([0]*n_b)

  del black
  del data1, data2, d1, d2
  del data10, data20, data11, data21, data12, data22
  
  return split_dataset(input_mri, output_mri, s)





