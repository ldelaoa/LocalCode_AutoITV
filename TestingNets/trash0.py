import os
import sys
import torch
import numpy as np
import csv
from skimage.measure import label, regionprops
from monai.metrics import DiceMetric,SurfaceDiceMetric,HausdorffDistanceMetric
from skimage.morphology import dilation,ball,erosion

background_bool = False
dice_metric = DiceMetric(include_background=background_bool, reduction="mean", get_not_nans=False)

#create random torch tensor
tensorlabel1 = torch.rand([1,288,288,197])
tensorlabel1[tensorlabel1>.5] = 1
tensorlabel1[tensorlabel1<=.5] = 0

tensorlabel2 = torch.rand([1,288,288,197])
tensorlabel2[tensorlabel2>.5] = 1
tensorlabel2[tensorlabel2<=.5] = 0

dice_metric(y_pred=tensorlabel1, y=tensorlabel2)
dice1 = dice_metric.aggregate().item()
print("Dice0 :'{:0.2f}".format(dice1))
dice_metric.reset()

lbl_3dnp = tensorlabel1.detach().cpu().numpy()
lbl_3dnp = lbl_3dnp.squeeze()
lbl_4d = np.expand_dims(lbl_3dnp, 0)
tensor_label = torch.from_numpy(lbl_4d)

output = tensorlabel2.detach().cpu().numpy()
output = output.squeeze()
output = np.expand_dims(output, 0)
tensor_output = torch.from_numpy(output)

dice_metric(y_pred=tensor_label, y=tensor_output)
dice1 = dice_metric.aggregate().item()
print("Dice0 :'{:0.2f}".format(dice1))
dice_metric.reset()