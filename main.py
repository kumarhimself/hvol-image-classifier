import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

img_size = 1024

def calc_euclidean_similarity(vec1, vec2):
    euc_dist = torch.norm(vec1 - vec2, p=2)

    n = vec1.shape[0]
    max_dist = n ** 0.5

    return 1 - (euc_dist / max_dist)

def img2vec(img_file_path):
    img = Image.open(img_file_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)

    pooled = F.adaptive_max_pool2d(img_tensor, output_size=(1, 1))
    print(f"Value of {img_file_path} pooled: {pooled.flatten()}")
    return pooled.flatten()

class_prototype = img2vec(os.path.join("data", "class_prototype.jpg"))

data = os.listdir("data")

for i in range(len(data)):
    if data[i].startswith("test"):
        current_test_vec = img2vec(os.path.join("data", data[i]))
        print(f"Similarity between class prototype and {data[i]}: {calc_euclidean_similarity(class_prototype, current_test_vec)}")
