import numpy as np


# Jaccard similarity for checking the performance of the model

def jaccard(tp, fp, fn):      # same as iou
  den = fp+tp+fn
  
  if den==0:
    return 1
  else:
    return round(tp/den, 2)




# Dice Score for checking the performance of the model

def dice_score(tp, fp, fn):         # same as f1-score
  den = fp+2*tp+fn

  if den==0:
    return 1
  else:
    return round(2*tp/den, 2)




# Precision for checking the performance of the model

def precision(tp, fp):
  den = tp+fp

  if den==0:
    return 1
  else:
    return round(tp/den, 2)




# Sensitivity measure for checking the performance of the model

def sensitivity(tp, fn):      # same as recall
  den = tp+fn

  if den==0:
    return 1
  else:
    return round(tp/den, 2)




# Accuracy for checking the performance of the model

def accuracy(tp, total):
  total_tp = np.sum(tp)
  return round(total_tp/total, 2)



