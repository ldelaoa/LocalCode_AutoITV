
import os
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
#from lungtumormask import mask as tumormask
from lungmask import mask as lungmask_fun
from skimage.measure import label, regionprops,shannon_entropy
from skimage.morphology import dilation,ball,erosion,remove_small_objects

from monai.transforms import (
    Compose,
    KeepLargestConnectedComponent,
    AsDiscrete,
    EnsureType,
    SaveImage,
    ScaleIntensity,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch,pad_list_data_collate

from CreateModel import CreateModel_fun,createVal_Loader_fun
from SaveFileMonai import saveFile_fun


def LookSortFiles(root_path, all_patientdir,gtv50_path):
    CTALL_fpaths =[]
    lungALL_fpaths =[]
    itv_fpaths = []
    gtv_fpaths = []    
    for patient_path in all_patientdir:
        ct_miss = 0
        gtv_miss = 0
        itv_miss = 0
        lung_miss = 0
        print(patient_path)
        # Look for the GTV of the 50%
        if gtv50_path is not None:
            for root, dirs, files in os.walk(gtv50_path + patient_path, topdown=False):
                for f in files:
                    if "_gtv" in f.lower():
                        gtv_fpaths.append(os.path.join(gtv50_path, patient_path, f))
                        gtv_miss +=1
        #Look for all CT of all BP
        for root, dirs, files in os.walk(root_path + patient_path, topdown=False):
            for f in files:
                if "_igtv" in f.lower() or "itv" in f.lower():
                    itv_fpaths.append(os.path.join(root_path, patient_path, f))
                    itv_miss +=1
                if "0%" in f.lower() and not("ave" in f.lower()):
                    if "ct" in f.lower():                            
                        CTALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        ct_miss +=1
                    if "lung" in f.lower():
                        lungALL_fpaths.append(os.path.join(root_path, patient_path, f))
                        lung_miss +=1

    print('ct: ',ct_miss,"Lungs: ",lung_miss,"GTV Miss: ", gtv_miss, "ITV Miss: ",itv_miss)
    if False:
        CreateLungMasks(root_path,CTALL_fpaths)
    
    CTALL_fpaths = np.sort(CTALL_fpaths)
    lungALL_fpaths = np.sort(lungALL_fpaths)
    itv_fpaths = np.sort(itv_fpaths)
    gtv_fpaths = np.sort(gtv_fpaths)
    return CTALL_fpaths, lungALL_fpaths


def EvalNet_fun(val_loader, modelSwin, modelDyn,image_size):
    out_channels = 2  # including background
    post_transforms = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, threshold=0.9),
            # FillHoles(applied_labels=1, connectivity=0),
            # RemoveSmallObjects(min_size=64, connectivity=3, independent_channels=True),
            ScaleIntensity(minv=0.0, maxv=1.0),
            KeepLargestConnectedComponent(is_onehot=False, connectivity=2, num_components=1),
        ]
    )
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, threshold=0.1)])

    # Testing the model
    print("Length val_loader: ", len(val_loader))

    predicted_GTV = []
    best_blob = False
    count = 0
    modelSwin.eval()
    modelDyn.eval()
    with torch.no_grad():
        for val_data in val_loader:

            val_inputs = val_data["image"].to(device)
            roi_size = image_size
            sw_batch_size = 1
            px = val_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-2]
            bp = val_data["image"].to('cpu').meta["filename_or_obj"][0].split('/')[-1].split('=')[-1]
            print('Px: ', 'BP: ', bp, " Count: ", px, count + 1)
            count += 1

            val_outputs_Swin = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelSwin)
            val_outPost_Swin = [post_transforms(i) for i in decollate_batch(val_outputs_Swin)]

            val_outputs_Dyn = sliding_window_inference(val_inputs, roi_size, sw_batch_size, modelDyn)
            val_outPost_Dyn = [post_transforms(i) for i in decollate_batch(val_outputs_Dyn)]

            val_outPost = torch.logical_or(val_outPost_Swin[0], val_outPost_Dyn[0])

            if best_blob:
                #tensor_blobn = best_blob_fun()
                #predicted_GTV.append(tensor_blobn)
                print("Import function")
            else:
                predicted_GTV.append(val_outPost)

            print("BP saved:", bp)
            path_to_save = os.path.join(figures_folder_i, px)
            FileToSave = val_outPost[0].detach().cpu().int()
            original_affine = val_data["image"].to('cpu').meta["affine"][0].numpy()
            nib.save(nib.Nifti1Image(FileToSave.astype(np.uint32), original_affine), os.path.join(path_to_save, bp+"_GTV_Predicted12.nii.gz"))

            print(1)
    return 0


def main(root_path,pretrained_path_Swin,pretrained_path_Dyn,figures_folder_i,name_run,gtv50_path):
    px_ = None
    if px_ is None:
        all_patientdir = []
        all_patientdir = os.listdir(root_path)
        all_patientdir.sort()
    else:
        all_patientdir = []
        all_patientdir.append(px_)
    print(len(all_patientdir),'in',name_run,all_patientdir)
    for px_dir in all_patientdir:
        templist = []
        templist.append(px_dir)
        CTALL_fpaths, lungALL_fpaths = LookSortFiles(root_path, templist,gtv50_path)

        data_dicts = [
            {"image": image_name,"lung":lung_name}
            for image_name,lung_name in zip(CTALL_fpaths,lungALL_fpaths)
        ]
        val_files =data_dicts[:]
        print('CT val len:',len(val_files))

        size = 96
        image_size = (size, size, size)

        val_loader = createVal_Loader_fun(cache,val_files,size)
        modelSwin,modelDyn = CreateModel_fun(size,device)

        # Load pretrained model
        if pretrained_path_Swin is not None:
            modelSwin.load_state_dict(torch.load(pretrained_path_Swin, map_location=torch.device(device)))
            print('Using Swin pretrained weights!')
        if pretrained_path_Dyn is not None:
            modelDyn.load_state_dict(torch.load(pretrained_path_Dyn, map_location=torch.device(device)))
            print('Using Dyn pretrained weights!')

        _ = EvalNet_fun(val_loader,modelSwin,modelDyn,image_size)

    return 0


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = False
    print('device:', device)
    gtv50_path =None#'/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti/'
    root_path = '/home/umcg/Desktop/AutomaticITV_code/SABR1322_Nifti/'

    preweight_path = '/home/umcg/Desktop/AutomaticITV_code/weights/v12/'
    pretrained_path_Swin = preweight_path + 'best_SwinUnet_V12_UMCG_TestSet3.pth'
    pretrained_path_Dyn = preweight_path + 'best_DynUnet_V12_UMCG_TestSet3.pth'

    #figures_path = '/home/umcg/Desktop/AutomaticITV_code/figures_folder_i/'
    #figures_folder_i = figures_path + 'figures_SwinDyn_V11_ITV/'
    figures_folder_i = root_path

    name_run = "TestRun" + "SwynDyn" + "ITV_FullContours" + "_V11"
    print(name_run)

    main(root_path, pretrained_path_Swin, pretrained_path_Dyn, figures_folder_i, name_run,gtv50_path)

    print("The end")
