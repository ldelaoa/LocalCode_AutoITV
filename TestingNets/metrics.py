import os
import sys
import torch
import numpy as np
import csv
from skimage.measure import label, regionprops
from monai.metrics import DiceMetric,SurfaceDiceMetric,HausdorffDistanceMetric
from skimage.morphology import dilation,ball,erosion


def tensor2tensor(tensor_label0,dilation_bool):
    tensorList = []
    propsList = []
    if len(tensor_label0[0].shape)==3:
        lbl_3dnp = tensor_label0.cpu().detach().numpy()
    else:
        lbl_3dnp = tensor_label0.cpu().detach().numpy()
        lbl_3dnp = lbl_3dnp.squeeze()
    if dilation_bool:
        lbl_3dnp = dilation(lbl_3dnp,ball(2))
    label_img = label(lbl_3dnp)
    propsL = regionprops(label_img)
    for r in range(len(propsL)):
        predicted_blobn = np.zeros(lbl_3dnp.shape)
        predicted_blobn[label_img == r + 1] = 1
        predicted_blobn = np.expand_dims(predicted_blobn, 0)
        tensor2 = [torch.from_numpy(predicted_blobn)]
        tensorList.append(tensor2)
        propsList.append(propsL[r])
    return tensorList,propsList


def metrics_fun(Metrics_tensor, tensor_label0,px,multiplier):
    background_bool = True
    dice_metric = DiceMetric(include_background=background_bool, reduction="mean", ignore_empty=False, get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=background_bool, reduction="mean", percentile=95)
    #surfDice_metric383 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 96*multiplier-1), include_background=background_bool)
    surfDice_metric191 = SurfaceDiceMetric(class_thresholds=np.linspace(3, 3, 192), include_background=background_bool)
    surfDice_metric287 = SurfaceDiceMetric(class_thresholds=np.linspace(6, 6, 288), include_background=background_bool)
    surfDice_metric383 = SurfaceDiceMetric(class_thresholds=np.linspace(9, 9, 384), include_background=background_bool)

    row_of_metrics = []

    if True:
        dice_metric(y_pred=Metrics_tensor, y=tensor_label0)  # [0][1:2, :, :, :] [0]
        dice0 = dice_metric.aggregate().item()
        print("Dice0 :'{:0.2f}".format(dice0))
        dice_metric.reset()

    label_tensor,propsLabel = tensor2tensor(tensor_label0,True)
    output_tensor, propsOutput = tensor2tensor(Metrics_tensor,True)

    for i in range(len(propsLabel)):
        rr = propsLabel[i]
        lbl_bbox = rr.bbox

    print("METRICS: blobs predicted: ", len(propsOutput))
    for n in range(len(propsOutput)):
        r = propsOutput[n]
        voxTP = 0
        #print(' bbox', r.bbox, "size", len(r.coords))

        TP = False
        for j in range(len(r.coords)):
            if (lbl_bbox[0] < r.coords[j][0] < lbl_bbox[3]) and (
                    lbl_bbox[1] < r.coords[j][1] < lbl_bbox[4]) and (
                    lbl_bbox[2] < r.coords[j][2] < lbl_bbox[5]):
                voxTP += 1
        print("ratio:",voxTP,rr.area,voxTP/rr.area)
        if (voxTP/rr.area) >.1:
            TP = True

        print('True Positive: ', TP)


        if TP:
            dice_metric(y_pred=output_tensor[n], y=label_tensor[0])
            dice1 = dice_metric.aggregate().item()
            print("Dice: '{:0.2f}".format(dice1))
            dice_metric.reset()

            hausdorff_metric(y_pred=output_tensor[n], y=label_tensor[0])
            hausd1 = hausdorff_metric.aggregate().item()
            hausdorff_metric.reset()
            print("Hausdorff: {:0.2f}".format(hausd1))

            if output_tensor[n][0].shape[1] == 288:
                surfDice_metric287(y_pred=output_tensor[n][0], y=label_tensor[0][0])
                sdice1 = surfDice_metric287.aggregate().item()
                surfDice_metric287.reset()
            elif output_tensor[n][0].shape[1] == 192:
                surfDice_metric191(y_pred=output_tensor[n][0], y=label_tensor[0][0])
                sdice1 = surfDice_metric191.aggregate().item()
                surfDice_metric191.reset()
            else:
                surfDice_metric383(y_pred=output_tensor[n][0], y=label_tensor[0][0])
                sdice1 = surfDice_metric383.aggregate().item()
                surfDice_metric383.reset()
            print('SurfDice :', sdice1)

            sumPredicted = torch.sum(output_tensor[n][0])
            sumGroundT = torch.sum(label_tensor[0][0])
            voxFP = abs(sumPredicted - voxTP)
            voxFN = abs(sumGroundT - voxTP)

            SegSens = voxTP / (voxTP + voxFP)
            SegPrec = voxTP / (voxTP + voxFN)
            print("Sensitivity: {:0.2f}".format(voxTP / (voxTP + voxFP)))
            print("Precision: {:0.2f}".format(voxTP / (voxTP + voxFN)))

            row_of_metrics.append([px,dice1,sdice1,hausd1,SegSens.item(),SegPrec.item(),TP,sumPredicted.item(),sumGroundT.item()])
        else:
            row_of_metrics.append([px, 0, 0, 0, 0, 0, False, 0, 0])
        print("----------------------------")
    return row_of_metrics


def saveMetrics_fun(path_to_save,file_to_save,all_metric_rows):
    print("Saving Metrics for Rows: ",len(all_metric_rows))
    f = open(path_to_save+file_to_save, 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(['Patient', 'Dice', 'SDice', 'HausD', 'SegSens','SegPRec','TP', 'AreaPredic', 'AreaLab'])
    for i in range(len(all_metric_rows)):
        writer.writerows(all_metric_rows[i])
    f.close()
    return 0
