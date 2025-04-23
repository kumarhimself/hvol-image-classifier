import torch
import torchvision.models as models
from PIL import Image

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final classification layer
_ = model.eval()

def img2veimg2vecc(img_tensor):
  if img_tensor.dim() == 3:
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
  with torch.no_grad():
    vec = model(img_tensor).squeeze()

  return vec
