import os
import glob
from PIL import Image
import cv2
import numpy as np
from train_dataset.data_config import get_dat_parent
filenames = os.listdir(get_dat_parent())

# 255,165,246,126,180,150,61,195,131,171,178,87,212,105
aallsit = []
for fold in filenames:
    if fold.endswith("_label"):
        train_origin_label = os.path.join(get_dat_parent()+fold)
        if not os.path.exists(get_dat_parent()+fold+"_new"):
            os.mkdir(get_dat_parent()+fold+"_new")
        train_origin_label_file = os.path.join(train_origin_label)
        all_mask= glob.glob(os.path.join(train_origin_label_file,"*.png"))
        for file in all_mask:
            img = Image.open(file)
            file_name = file.split("\\")[-1]
            #print(np.array((img)))
            for nn in np.array((img)):
                for n in nn:
                    if n!=0 and n not in aallsit:
                        aallsit.append(n)
                        print(n)
            #break
            # print(file_name)
            # new = Image.new("RGB", [np.shape(img)[1], np.shape(img)[0]])
            # new = new + np.expand_dims(0 * (np.array(img) == 0), -1)
            # new = new + np.expand_dims(1 * (np.array(img) == 255), -1)
            # new = new + np.expand_dims(2 * (np.array(img) == 246), -1)
            # new = new + np.expand_dims(3 * (np.array(img) == 180), -1)
            # new = new + np.expand_dims(4 * (np.array(img) == 165), -1)
            # new = new + np.expand_dims(5 * (np.array(img) == 150), -1)
            # new = new + np.expand_dims(6 * (np.array(img) == 61), -1)
            # new = new + np.expand_dims(7 * (np.array(img) == 195), -1)
            # new = Image.fromarray(np.uint8(new))
            # #print(np.max(new),np.min(new))
            # #new.show()
            # new.save(os.path.join(get_dat_parent()+fold+"_new",file_name))