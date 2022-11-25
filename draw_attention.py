import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
import os
import matplotlib
import threading

from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

base = "/mnt21t/home/mm2022/fast-reid/logs/MarketT/bap_R50_lupNL/attn/"
attention_path = base + "raw_attention/"
image_path = base + "raw_image/"

heatmap_path = base + "heatmap/"

# Values to be used for image normalization
mean = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
std = [0.229*255, 0.224*255, 0.225*255]

data = {}

for maindir, subdir, file_name_list in os.walk(attention_path):
    for filename in file_name_list:        
        data.setdefault(filename,{})["attention"] = np.load(maindir+filename)

for maindir, subdir, file_name_list in os.walk(image_path):
    for filename in file_name_list:
        data.setdefault(filename,{})["image"] = np.load(maindir+filename)

for k, v in data.items():
    attention = v["attention"]
    image = v["image"]

    attention_maps = F.upsample_bilinear(torch.Tensor([attention]), size=(256, 128))
    attention_maps = torch.sqrt(attention_maps / attention_maps.max().item())[0]   
    attention_mean = torch.max(attention_maps, 0, keepdim=True).values
    attention_maps = torch.cat((attention_maps, attention_mean), dim=0)
    data[k]["attention"] = attention_maps.numpy()
    
    raw_image = image.transpose((1,2,0)) * std + mean
    data[k]["image"] = raw_image

def output_attention(image, attention, file_path):
    s = np.array(image, dtype=int)

    plt.imshow(s, alpha = 1)
    plt.imshow(attention, cmap ="jet", alpha = 0.6)

    plt.axis(False)
    plt.savefig(file_path)

def save_image(image, image_path):
    s = np.array(image, dtype=int)
    plt.imshow(s, alpha = 1)
    plt.axis(False)
    plt.savefig(image_path)

for k, v in tqdm(data.items()):
    attention = v["attention"]
    image = v["image"]

    dir = heatmap_path+k.split(".")[0]+"/"

    for i in range(attention.shape[0]):
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_attention(image, attention[i], dir+str(i)+".png")
    
    save_image(image, dir+"raw.png")

