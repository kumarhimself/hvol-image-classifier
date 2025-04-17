import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

def img2vec(img_tensor):
    binarized_img = (img_tensor >= 0.5).float()
    binarized_img = F.avg_pool2d(binarized_img, kernel_size=16, stride=16)

    return binarized_img.flatten()
