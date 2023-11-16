import glob
import cv2
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

dirs = {
    'low_res': 'data/low_res',
    'high_res': 'data/source',
    'metrics': 'data/metrics.xlsx'
}

def load_case(case_name, preprocess=False, input_shape=(512,512)):
    case_num = case_name.split('_')[-1]

    path_list = glob.glob(f"{dirs['high_res']}/{case_num}.nii.gz")
    data_im = sitk.ReadImage(path_list[0])
    target = sitk.GetArrayFromImage(data_im)

    path_listp = glob.glob(f"{dirs['low_res']}/{case_num}.nii.gz")
    datap_im = sitk.ReadImage(path_listp[0])
    inp = sitk.GetArrayFromImage(datap_im)

    if preprocess:

        X1 = []
        X2 = []

        for i in range(len(inp)):
            x1 = np.expand_dims(cv2.resize(inp[i], input_shape, interpolation = cv2.INTER_NEAREST),axis=-1).astype('float32')
            x2 = np.expand_dims(cv2.resize(target[i], input_shape, interpolation = cv2.INTER_NEAREST),axis=-1).astype('float32')
            X1.append(x1)
            X2.append(x2)

        return normalize(np.array(X1)), normalize(np.array(X2))

    return inp, target

def normalize(im):
    im = (((im - im.min())/(im.max()-im.min()))*2)-1
    return im

def DataLoader(case_list, shape=(512,512)):

    inp_img = []
    tar_img = []

    for case_name in tqdm(case_list):
        X1, X2= load_case(case_name)

        X1 = normalize(X1)
        X2 = normalize(X2)

        for i in range(len(X1)):
            inp = np.expand_dims(cv2.resize(X1[i], shape, interpolation = cv2.INTER_NEAREST),axis=-1).astype('float32')
            target = np.expand_dims(cv2.resize(X2[i], shape, interpolation = cv2.INTER_NEAREST),axis=-1).astype('float32')
            inp_img.append(inp)
            tar_img.append(target)

    inp_img = np.array(inp_img)
    tar_img = np.array(tar_img)
    train_dataset = [inp_img, tar_img]

    return train_dataset
