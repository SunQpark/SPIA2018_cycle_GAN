import sys, os
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from crop_face import CropFace
from util import *

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp')

pad_nearest = transforms.Pad(50, padding_mode='edge')

attr_path = '../data/CelebA/list_attr_celeba.txt'
attr = pd.read_csv(attr_path, delim_whitespace=True, usecols=['Smiling']) > 0

## to align sketch images
# original_path = '../data/sketch_images/human'
# original_path = '../data/sketch_images/sketch'
# original_path = '../data/sketch_images/sketches'
# target_path = '../data/aligned_new/sketch'

## to align real images

original_path = '../data/CelebA/img/img_align_celeba'
target_path = '../data/aligned_new/picture'
# target_path = '../data/aligned_new/sketch/mod_celeba'
ensure_dir(target_path)


# setup for landmark detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

count = 0
for i, fname in enumerate(tqdm(attr.index)):
# for i, fname in enumerate(tqdm(os.listdir(original_path))):
    if attr['Smiling'].loc[fname]:
        if np.random.rand() < 0.95:
            continue

    file_path = os.path.join(original_path, fname)

    img = dlib.load_rgb_image(file_path) # img: numpy array (h, w, c)
    img_pil = Image.fromarray(img)


    # # applying sketch effect filter with random 
    # radius = 0.9 + 0.5 * np.random.rand() 
    # bright = 1.6 + 0.4 * np.random.rand()
    # contrast = 1.6 + 1.4 * np.random.rand()
    # img_pil = sketch_filter(img_pil, radius, bright, contrast)

    
    dets = detector(img, 2) # TODO: try other numbers
    
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        l_eye = (shape.part(36).x, shape.part(36).y)
        r_eye = (shape.part(45).x, shape.part(45).y)

        if shape.part(45).x - shape.part(36).x < 20:
            continue

        count += 1
        out_name = f'{i:06}.png'
        CropFace(img_pil, eye_left=l_eye, eye_right=r_eye, offset_pct=(0.3,0.3), dest_sz=(128,128)).save(os.path.join(target_path, out_name))
    
    if count >= 1500:
        break

