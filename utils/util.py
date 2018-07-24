import os
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
from PIL.ImageOps import invert
from PIL.ImageFilter import GaussianBlur


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sketch_filter(img, blur_radius=0.8, brightness_ratio=2, contrast_ratio=2):
    """ 
    generate sketch images by applying filters to the photograph images.
    tested range of parameters:
        blur_radius - 0.8~1.5
        brightness_offset - 1.5~2.0
        contrast_offset - 1.5~3.0
    """
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    img_tensor_orig = to_tensor(img)
    img_blur = img.filter(GaussianBlur(blur_radius))
    img_tensor_blur = to_tensor(img_blur)

    img_mean = (1 + img_tensor_orig - img_tensor_blur) / 2

    img = to_pil(img_mean)

    img = ImageEnhance.Brightness(img).enhance(brightness_ratio)
    img = ImageEnhance.Contrast(img).enhance(contrast_ratio)
    
    return img


if __name__ == '__main__':
    # test: sketch filter
    filename = '000001.jpg'
    img_dir = '../data/CelebA/img/img_align_celeba/'
    img_path = os.path.join(img_dir, filename)
    img_orig = Image.open(img_path)

    img_result = sketch_filter(img_orig)
    img_result.show()
    
    