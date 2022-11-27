
import argparse
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

def output_attention(image, attention=None, file_path=None):
    s = np.array(image, dtype=int)
    plt.axis(False)
    plt.imshow(s, alpha = 1)
    
    if attention is not None:
        # make it easy to see
        attention = attention * attention
        
        plt.imshow(attention, cmap ="jet", alpha = 0.5)
    
    # plt.show()
    plt.savefig(file_path)

def save_image(image, image_path):
    s = np.array(image, dtype=int)
    plt.imshow(s, alpha = 1)
    plt.axis(False)
    plt.savefig(image_path)

def normalize(a):
    shape = a.shape
    b = torch.tensor(a).flatten()
    a_std = torch.std(b, axis=-1)
    a_mean = torch.mean(b, axis=-1)
    b = (b-a_mean)/a_std
    return b.view(shape).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, help='path to the attention map')
    base = parser.parse_args().base

    attention_path = os.path.join(base, "raw_attention")
    image_path = os.path.join(base, "raw_image")
    heatmap_path = os.path.join(base, "heatmap")

    if not os.path.isdir(base):
        os.mkdir(base)
        
    if not os.path.isdir(heatmap_path):
        os.mkdir(heatmap_path)

    # Values to be used for image normalization
    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    data = {}

    for maindir, subdir, file_name_list in os.walk(image_path):
        for filename in tqdm(file_name_list):
            strs = filename.split(".")
            data.setdefault(strs[0],{})["image"] = np.load(os.path.join(maindir, filename))

    for maindir, subdir, file_name_list in os.walk(attention_path):
        for filename in tqdm(file_name_list): # 0479_c5s1_125520_00_a3.npy -> key: 0479_c5s1_125520_00  attn_3.npy  
            strs = filename.split("_a")

            if strs[0] not in data:
                data[strs[0]] = {}
            if "attention" not in data[strs[0]]:
                data[strs[0]]["attention"] = {}

            data[strs[0]]["attention"]["attn_"+strs[-1]] = np.load(os.path.join(maindir, filename))

    for k, v in tqdm(data.items()):       
        def scale(attention):

            attention_maps = F.upsample_bilinear(torch.Tensor([attention]), size=(384, 192))[0]
            attention0 = attention_maps[0].view(1, 384, 192).permute((1,2,0)).numpy() # single
            attention1 = attention_maps[1].view(1, 384, 192).permute((1,2,0)).numpy() # single
            attention_mean = torch.mean(attention_maps, dim=0, keepdim=True).permute((1,2,0)).numpy()

            return [attention0, attention1, attention_mean]

        data[k]["attention"]["a4"] = scale(data[k]["attention"]["attn_4.npy"])
        data[k]["attention"]["a3"] = scale(data[k]["attention"]["attn_3.npy"])
        
        image = v["image"]
        raw_image = image.transpose((1,2,0)) * std + mean
        data[k]["raw_image"] = raw_image

    if not os.path.isdir(heatmap_path):
        os.mkdir(heatmap_path)

    for k, v in tqdm(data.items()):
        attention = v["attention"]
        image = v["raw_image"]

        dir = os.path.join(heatmap_path, k)

        if not os.path.isdir(dir):
            os.mkdir(dir)

        output_attention(image, file_path=os.path.join(dir, "raw.png"))

        for i in range(3):
            output_attention(image, attention["a3"][i], os.path.join(dir, "a3_"+str(i)+".png"))
            output_attention(image, attention["a4"][i], os.path.join(dir, "a4_"+str(i)+".png"))
