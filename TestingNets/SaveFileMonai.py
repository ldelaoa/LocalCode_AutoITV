from monai.transforms import (
    SaveImage,
)


def saveFile_fun(FileToSave,path_to_save,Filename):
    trans = SaveImage(
        output_dir=path_to_save,
        output_postfix=Filename,
        output_ext='.nii.gz',
        resample=True,
        separate_folder=False,  #False if saving into the same folder
    )
    trans(FileToSave)
    return 0