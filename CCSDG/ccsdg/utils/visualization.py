import numpy as np
import nibabel as nib


def visualization_as_nii(numpy_arr, saved_name):
    new_image = nib.Nifti1Image(numpy_arr, np.eye(4))
    new_image.set_data_dtype(np.float32)
    nib.save(new_image, saved_name)


def normalize_img_to_0255(img):
    return (img-img.min())/(img.max()-img.min()) * 255
