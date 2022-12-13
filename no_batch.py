import shutil, os, sys
from pathlib import Path
import numpy as np
import json, math
from cv2 import imread, imwrite, IMREAD_GRAYSCALE
from multiprocessing import Pool

f = open('no_tie.json')
order = json.load(f)
mapping = open('w2i_aav2.json')
w2i = json.load(mapping)
# Sys.argv[1] = In folder
# Sys.argv[2] = out folder
sorted_order = list(sorted(order.items(), key=lambda item: item[1], reverse=True))
Path(f'{sys.argv[2]}/x/').mkdir(parents=True, exist_ok=True)
Path(f'{sys.argv[2]}/y/').mkdir(parents=True, exist_ok=True)

def convert(wav):
    pad_size = 512
    if wav[1] <= pad_size:
        gr = np.zeros(shape=(128, pad_size))
        name = wav[0].split('.')[0]
        loaded_img = imread(f'{sys.argv[1]}/{name}.png', IMREAD_GRAYSCALE)
        gr[:loaded_img.shape[0],:loaded_img.shape[1]] = loaded_img
        f_in = open(f'/mnt/c/users/trifo/desktop/MUMT 203 AMT/package_aa/{name}/{name}.semantic', 'r')
        notes = [w2i[note] for note in f_in.read().split() if not ('barline' in note or 'keySignature' in note or 'clef' in note or 'timeSignature' in note)]
        np.save(f'{sys.argv[2]}/y/{name}.npy', notes)
        imwrite(f'{sys.argv[2]}/x/{name}.png', gr)
        
        return name

with Pool(10) as p:
    valid_list = p.map(convert, sorted_order)

print(len(valid_list))
# np.save('valid_list.npy', valid_list)




    