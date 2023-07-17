import os
import numpy as np
from glob import glob
import cv2
from natsort import natsorted
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
import torch
import utils
from PIL import Image

def read_img(path):
    return Image.open(path)


def aug_img(aug, inp_img, tar_img):
    # inp_img = TF.to_tensor(inp_img)
    # tar_img = TF.to_tensor(tar_img)
    if aug == 1:
        inp_img = np.flip(inp_img,1)
        tar_img = np.flip(tar_img,1)
    elif aug==2:
        inp_img = np.flip(inp_img,0)
        tar_img = np.flip(tar_img,0)
    elif aug==3:
        inp_img = np.rot90(inp_img)
        tar_img = np.rot90(tar_img)
    elif aug==4:
        inp_img = np.rot90(inp_img, k=2)
        tar_img = np.rot90(tar_img, k=2)
    elif aug==5:
        inp_img = np.rot90(inp_img, k=3)
        tar_img = np.rot90(tar_img, k=3)
    elif aug==6:
        inp_img = np.rot90(np.flip(inp_img,1))
        tar_img = np.rot90(np.flip(tar_img,1))
    elif aug==0:
        inp_img = np.rot90( np.flip(inp_img,0))
        tar_img = np.rot90(np.flip(tar_img,0))
    
    return inp_img, tar_img



def main():
    datasets = {'GoPr', 'HIDE'};
    file_path = os.path.join('Datasets/GoPr/train', 'input')
    gt_path = os.path.join('Datasets/GoPr/train', 'target')
    save_path = "tes/input/"
    save_path_t = "tes//target/"
    print(file_path)
    print(gt_path)

    path_fake = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    path_real = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    print(len(path_fake))


    for i in range(len(path_real)):
        t1 = read_img(path_real[i])
        t2 = read_img(path_fake[i])
        for j in range(7):
            inp_img, tar_img = aug_img(j,t2, t1)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path_t, exist_ok=True)
            inp_img_path = save_path+str(j)+'go'+str(i)+'.png'
            tar_img_path = save_path_t+str(j)+'go'+str(i)+'.png'
            utils.save_img(inp_img_path, inp_img)
            utils.save_img(tar_img_path, tar_img)



   

if __name__ == '__main__':
    main()


 
 
 
 