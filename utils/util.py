import os
import torch
from torch.autograd import grad
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

# code ref: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def calc_gradient_penalty(netD, real_data, fake_data, lambd=10):
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.shape)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones_like(disc_interpolates), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
    return gradient_penalty

if __name__ == '__main__':
    # test: sketch filter
    filename = '000001.jpg'
    img_dir = '../data/CelebA/img/img_align_celeba/'
    img_path = os.path.join(img_dir, filename)
    img_orig = Image.open(img_path)

    img_result = sketch_filter(img_orig)
    img_result.show()
    
    