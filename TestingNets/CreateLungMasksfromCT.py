import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
#from lungtumormask import mask as tumormask
from lungmask import mask as lungmask_fun


def CreateLungMasks(root_path, CT_fpaths):
    # Get Lung mask and save it
    CT_path0 = CT_fpaths[0]
    CT_nii = nib.load(CT_path0)
    for ct in CT_fpaths:
        empty_header = nib.Nifti1Header()
        lung_path = ct[:-10] + '_LungMask.nii.gz'
        print('Creating Lung Mask: ', lung_path)
        input_image = sitk.ReadImage(ct, imageIO='NiftiImageIO')
        lungmask = lungmask_fun.apply(input_image)  # default model is U-net(R231)
        lungmask_ni = nib.Nifti1Image(lungmask, CT_nii.affine, empty_header)
        nib.save(lungmask_ni, lung_path)
    return 0
