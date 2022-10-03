import shutil, os, sys
from pathlib import Path
import numpy as np
import json, math
from skimage.io import imread, imsave

f = open('test_order.json')
order = json.load(f)

sorted_order = list(sorted(order.items(), key=lambda item: item[1], reverse=True))
print(sorted_order)
batch_size = 8
batch_count = 1
for idx in range(0, len(sorted_order), batch_size):
    batch = sorted_order[idx : idx + batch_size]
    pad_size = batch[0][1]
    gr_chunk = np.zeros(shape=(len(batch), 128, pad_size))
    Path(f'{sys.argv[2]}/x/batch_{batch_count}').mkdir(parents=True, exist_ok=True)
    Path(f'{sys.argv[2]}/y/batch_{batch_count}').mkdir(parents=True, exist_ok=True)
    for n, img in enumerate(batch):
        gr = np.zeros(shape=(128, pad_size))
        name = img[0].split('.')[0]
        loaded_img = imread(f'{sys.argv[1]}/{name}.png')
        gr[:loaded_img.shape[0],:loaded_img.shape[1]] = loaded_img
        shutil.copy(f'/mnt/c/users/trifo/desktop/MUMT 203 AMT/package_aa/{name}/{name}.semantic', f'{sys.argv[2]}/y/batch_{batch_count}/{name}.semantic')
        imsave(f'{sys.argv[2]}/x/batch_{batch_count}/{name}.png', gr)
        # shutil.move(f'{sys.argv[1]}/{name}.png', f'{sys.argv[2]}/x/batch_{batch_count}/{name}.png')
        gr_chunk[n][:loaded_img.shape[0],:loaded_img.shape[1]] = loaded_img
    imsave(f'{sys.argv[2]}/x/batch_{batch_count}/batch_{batch_count}.png', gr_chunk.reshape((len(batch)* 128, pad_size)))
    np.save(f'{sys.argv[2]}/x/batch_{batch_count}/batch_{batch_count}.npy', gr_chunk)
    batch_count += 1




    