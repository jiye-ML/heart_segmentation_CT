from nets.unet import mobilenet_unet
from PIL import Image
import numpy as np
import random
import copy
import os
import glob

random.seed(0)
NCLASSES = 15
class_colors = [[random.randint(200, 255), random.randint(200, 255), random.randint(0, 255)] for _ in range(NCLASSES)]
HEIGHT = 416
WIDTH = 416

model = mobilenet_unet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/ep056-loss0.242-val_loss0.215.h5")

# imgs = []
# with open("train_dataset/test_data.txt", "r") as f:
#     fff = f.readlines()
#     for img in fff:
#         imgs.append(img.split(";")[0].strip())
# np.random.shuffle(imgs)

test_img = './img_real'
imgs = glob.glob(os.path.join(test_img, "*.png"))
LABEL = [0, 255, 165, 246, 126, 180, 150, 61, 195, 131, 171, 178, 87, 212, 105]
for jpg in imgs:

    img = Image.open(jpg)
    # old_img = copy.deepcopy(img)
    # old_img = np.array(old_img)
    # old_img = np.expand_dims(old_img, axis=-1)
    # old_img = np.repeat(old_img, repeats=3, axis=-1)
    # old_img = Image.fromarray(old_img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]
    print(jpg, orininal_h, orininal_w)
    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, repeats=3, axis=-1)
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    pr = model.predict(img)[0]
    #
    pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)
    # for im in np.array(pr):
    #     for mm in im:
    #         if mm != 0:
    #             print(mm)
    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2)))
    colors = class_colors
    seg_img[:, :] = pr[:, :]
    for c in range(NCLASSES):
        seg_img[:, :] += ((pr[:, :] == c) * (LABEL[c])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h)).convert('RGB')

    jpg_name = os.path.basename(jpg)
    seg_img.save("./img_out/" + jpg_name)
    # image = Image.blend(old_img,seg_img,0.3)
    # jpg_name = jpg.split("/")[-1]
    # image.save("./img_out/"+jpg_name)
