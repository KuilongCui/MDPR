import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import matplotlib
from tqdm import *
import math

base_dir = '/mnt21t/home/mm2022/fast-reid/logs/market1501/bagtricks_R50/'
bagtricks_dir = base_dir + 'mismatch/'

if not os.path.isdir(bagtricks_dir):
    os.mkdir(bagtricks_dir)

def getMethodInfo(filepath):
    data = np.load(filepath, allow_pickle = True)
    return data
    
def sum_list(l, c=19):
    s = 0
    for i, val in enumerate(l):
        if i < c:
            s += val
    return s


bagtricks_filepath = '/mnt21t/home/mm2022/fast-reid/logs/market1501/bagtricks_R50/mismatch.npy'
bagtricks_data = getMethodInfo(bagtricks_filepath)

def PrintSingleQuery(data, save_dir):
    query_path =  data[0]
    query_id = query_path.split('/')[-1].split('_')[0]

    results = data[1]
    gallery = results.split()
    match = data[2]
    score = sum_list(match)
    
    fig=plt.figure(dpi=500,figsize=(15,15))
    fig.tight_layout()

    for j in range(1,len(gallery)):
        temp = fig.add_subplot(1, len(gallery), j)
        temp.set_xticks([])
        temp.set_yticks([])

        if match[j-1] == 0:
            temp.patch.set_linewidth(6)
            temp.patch.set_edgecolor('yellow')

        img_g = cv2.imread(gallery[j-1])
        img_g = cv2.resize(img_g, (128, 256), interpolation=cv2.INTER_CUBIC)
        img_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2RGB)
        temp.imshow(img_g)
        
    plt.savefig(save_dir + '{}.png'.format(query_id))
    plt.close(fig)

    return query_id, score

bagtricks_score = {}

for i in tqdm(range(bagtricks_data.shape[0])):
    query_id, score = PrintSingleQuery(bagtricks_data[i], bagtricks_dir)
    bagtricks_score[query_id] = score

    if i > 40:
        break

# mean = np.mean(bagtricks_score.values())
# for k,v in bagtricks_score.items():
#     if v < mean:
#         print(k)