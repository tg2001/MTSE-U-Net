import numpy as np
from Testing.evaluation_metrics import *
from tqdm import tqdm
import time





'''
Function for getting the class in which, each pixel of the output belongs, and is stored in 'req' varible.

(The model returns a list of 8 values for each pixel (the probability of pixel belonging to each class,
as the segmentation task was treated as a classication task for each pixel and there are 8 classes in total)

The function then calculates the true positive, false positive and false negative count for each class if the outputs are passed to actual
(by comparing the new output with the ground truth), which are returned along with the new output

'actual' has a default value of [] to indicate that the 'req' variable is to be returned
'''

def get_count(output, actual=[]):
  
  req = np.zeros((256, 256), dtype=np.uint8)    # Variable in which the new output will be stored
  out = output[0].reshape((256, 256, 8))

  tp = np.zeros(8)
  fp = np.zeros(8)
  fn = np.zeros(8)

  for i in range(256):
    for j in range(256):

      a = out[i, j, :].tolist()     # For each pixel, the 8 probability values are stored in the list a
      req[i, j] = a.index(max(a))   # Returning the index of the list whose value (probability of belonging to that class) is maximum
      
      if list(actual):
        if req[i, j]==actual[i, j]:
          tp[req[i, j]] += 1

        elif (not req[i, j]==actual[i, j]):

          fp[req[i, j]] += 1
          fn[actual[i, j]] += 1
  
  if not list(actual):
    return req

  return tp, fp, fn, req







# Function for calculating the evaluation metrics for each class 
# (using the true positive, false positive and false negative count for each class) 
# Then either those metrics are returned directly, or the average of those metrics, along with the accuracy, are calculated and returned,
# depending on the value of k

def calc_met(tp, fp, fn, total, k):

  prec = []
  dice = []
  jac = []
  sens = []

  for i in range(8):
    prec.append(precision(tp[i], fp[i]))
    sens.append(sensitivity(tp[i], fn[i]))
    dice.append(dice_score(tp[i], fp[i], fn[i]))
    jac.append(jaccard(tp[i], fp[i], fn[i]))

  if k==1:
    return prec, sens, jac, dice

  avg_prec = round(sum(prec)/8, 2)
  avg_dice = round(sum(dice)/8, 2)
  avg_jac = round(sum(jac)/8, 2)
  avg_sens = round(sum(sens)/8, 2)
  acc = round(accuracy(sum(tp), total), 2)

  if k==0:
    return avg_prec, avg_sens, avg_jac, avg_dice, acc
  
  
  
  
  
  
  
  
  # Function for printing the mean and the standard deviation of the average of the evaluation metric 
# results for all the classes over all the images

def cal_avg_metric(metrics):

  c = 1
  for i in metrics:

    i_mean = round(np.mean(i), 2)
    i_std = round(np.std(i), 2)

    print('\nMetrics no.', c, 'for the average of all brain parts')
    print(i_mean, i_std)
    c += 1
  
  
  

  
  
  # Function for printing the mean and the standard deviation of all the evaluation metric results for all the classes over all the images

def cal_all_metric(metrics, num):
  # metrics = [prec_list, sens_list, jac_list, dice_list]

  c = 1
  for k in metrics:
    print('\nMetrics no.', c, 'for all brain parts\n')
    c += 1

    for i in range(1, 8):
      l = []

      for j in range(num):
        l.append(k[j][i])

      l_mean = round(np.mean(l), 2)
      l_std = round(np.std(l), 2)
      print(l_mean, l_std)
      
      
      
      
      
      
      
      
# This function calculates the prediction accuracy of the fetal mri type and age for the combined task
# This function can also return the actual classes for type and age prediction of the combined task, 
# when nothing is passed to the parameter 'y_test' and the default value is used

def type_and_age_eval(out2, out3, num, y_test=[]):

  count = 0
  age_err = []

  for k in range(num):
    if list(y_test):
      if out2[k]==y_test[1][k]:
        count += 1

    else:
      if out2[k]==0:
        out2[k] = 'Type cannot be determined from the slice'
      elif out2[k]==1:
        out2[k] = 'Pathological'
      else:
        out2[k] = 'Neurotypical'

    if list(y_test):
      y = y_test[2][k]

      if y>0:
        out3[k] = (out3[k]*10)+15
        y = (y*10)+15

      age_err.append(abs(out3[k]-y))

    else:
      if out3[k]<0.4:
        out3[k] = 'Age cannot be determined from the slice'
      else:
        out3[k] = round((out3[k]*10)+15, 1)

  if not list(y_test):
    return out2, out3

  accu = round(count/num, 2)

  print('\nAccuracy for type predition:', accu)

  if num==1:
    print('Error for age prediction task:', round(age_err[0], 1))
          
  else:
    err_avg = round(np.mean(age_err), 2)
    err_sd = round(np.std(age_err), 2)
    print('Error for age prediction task:', err_avg, err_sd)
    
    
    
    
    
    
'''
Function for prediction and evaluation for the combined task 
(The combined task is the task of segmentation and prediction of fetal type and age from the mri by a single model)

For each input image, the segmentation and the type and age prediction are performed simultaneously.

After segmentation, the output passed on to the two functions:
'get_count' for getting the actual output, along with the true positive, false positive and false negative, and
'calc_met' for finally getting the evaluation metric values or average values (determined by the 'all' value) per image.

The values returned by calc_met are added to a list for getting the mean and SD of those values over all the test images by:
'cal_avg_metric', for all=0 or 'cal_all_metric', for all=1

The predicted type and age values are stored in lists out2 and out3, and at last, 
the function 'type_and_age_eval' is called for evaluating the predictions

This process is repeated for all the test images to be evaluated

This function can also return only the predicted values, instead of the metric values, in which case, 
nothing is passed to variables 'y_test' and 'all', so they are initialised to their default values
In that case, the 'type_and_age_eval' can return the actual class names for type and age prediction for the combined task
'''

def pred_and_eval(model, X_test, y_test=[], all=0):

  if len(X_test.shape)<3:       # For a single test case, converting the 2D slice to a 3D array
    X_test = [X_test]

    if list(y_test):
        y_test = [[y_test[0]], [y_test[1]], [y_test[2]]]

  prec_list = []
  dice_list = []
  jac_list = []
  sens_list = []
  acc_list = []

  out2 = []
  out3 = []

  new_out = []
  num = len(X_test)   # Number of test cases

  for k in tqdm(range(num), desc="Executing", ncols=75):

    output = model.predict(X_test[k].reshape(1, 256, 256))

    o2 = list(output[1][0])
    o2 = o2.index(max(o2))        
    out2.append(o2)

    o3 = output[2][0][0]
    out3.append(o3)

    output = output[0]

    if list(y_test):
      actual = y_test[0][k]
      tp, fp, fn, out = get_count(output, actual)

      if all==0:
        prec, sens, jac, dice, acc = calc_met(tp, fp, fn, 256*256, all)
        acc_list.append(acc)

      elif all==1:
        prec, sens, jac, dice = calc_met(tp, fp, fn, 256*256, all)

      prec_list.append(prec)
      dice_list.append(dice)
      jac_list.append(jac)
      sens_list.append(sens)

    else:
      new_out.append(get_count(output))

  print()

  if not list(y_test):
      out2, out3 = type_and_age_eval(out2, out3, num)
      return new_out, out2, out3

  elif num==1:                          # For num = 1, the metric values will be displayed directly, instead of the mean and SD, since they store only a single value
    print('Metric values are: ')

    print(prec_list[0])
    print(sens_list[0])
    print(jac_list[0])
    print(dice_list[0])

    if all==0:
      print(acc_list[0])
    
  elif all==0:
    cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])

  elif all==1:
    cal_all_metric([prec_list, sens_list, jac_list, dice_list], num)

  if list(y_test):
    type_and_age_eval(out2, out3, num, y_test)
  
  else:
    return type_and_age_eval(out2, out3, num)
  
  
  
  
