import os
from train_dataset.data_config import get_dat_parent
import numpy as np
import glob

allimgs =[]
allabel = []
with open("tmp.txt","w") as f:
    for fold in range(1,21):
         if fold<10:
             ind = "0"
         else:
             ind =""
         allabel = glob.glob(get_dat_parent() + "ct_train_10"+ind+str(fold)+"_label" + "/*.png")
         print(len(allabel))
         for index in range(0,len(allabel)):
             f.write(get_dat_parent() + "ct_train_10"+ind+str(fold)+"_image" + "/"+str(index)+".png"+";"+
                     get_dat_parent() + "ct_train_10"+ind+str(fold)+"_label" + "/"+str(index)+".png")
             f.write("\n")

k = 0
train = open("train_data.txt", "w")
test = open("test_data.txt", "w")
with open("tmp.txt","r") as f:
    allfile = f.readlines()
    np.random.shuffle(allfile)
    for ii in allfile:
        if k<300:
            test.write(ii)
        else:
            train.write(ii)
        k+=1

train.close()
test.close()