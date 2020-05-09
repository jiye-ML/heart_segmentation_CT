import shutil
import os
from train_dataset.data_config import  get_dat_parent
rootpath = get_dat_parent()
test_data ="../img_real" #rootpath+"test_dataset"
if not os.path.exists(test_data):
    os.mkdir(test_data)

with open("test_data.txt","r") as f:
    for file in f.readlines():
        origin_path = file.split(";")[-1].strip()
        origin_name = origin_path.split("/")[-1]
        shutil.copy(origin_path, test_data + "/" + origin_name)
