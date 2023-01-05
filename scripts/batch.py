import shutil, os, sys
from pathlib import Path
import numpy as np
import json, math
from cv2 import imread, imwrite, IMREAD_GRAYSCALE

order_aa = json.load(open('no_tie_aa.json'))
order_ab = json.load(open('no_tie_ab.json'))
combined_order = order_aa | order_ab
mapping = open('w2i_all.json')
w2i = json.load(mapping)
# Sys.argv[1] = In folder
# Sys.argv[2] = out folder
sorted_order = list(sorted(combined_order.items(), key=lambda item: item[1], reverse=True))
# print(sorted_order)
batch_size = 16
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
        loaded_img = imread(f'{sys.argv[1]}/{name}.png', IMREAD_GRAYSCALE)
        gr[:loaded_img.shape[0],:loaded_img.shape[1]] = loaded_img
        # shutil.copy(f'/mnt/c/users/trifo/desktop/MUMT 203 AMT/package_aa/{name}/{name}.semantic', f'{sys.argv[2]}/y/batch_{batch_count}/{name}.semantic')
        # print(img)
        if img[0] in order_aa:
            f_in = open(f'/mnt/c/users/trifo/desktop/MUMT 203 AMT/package_aa/{name}/{name}.semantic', 'r')
        else:
            f_in = open(f'/mnt/c/users/trifo/desktop/MUMT 203 AMT/package_ab/{name}/{name}.semantic', 'r')
        notes = [w2i[note] for note in f_in.read().split() if not ('barline' in note or 'keySignature' in note or 'clef' in note or 'timeSignature' in note)]
        # notes_mapped = [w2i[note] for note in notes]
        # f_out = open(f'{sys.argv[2]}/y/batch_{batch_count}/{name}.semantic', 'w')
        # f_out.write(' '.join(notes))
        # f_out.close()
        np.save(f'{sys.argv[2]}/y/batch_{batch_count}/{name}.npy', notes)
        imwrite(f'{sys.argv[2]}/x/batch_{batch_count}/{name}.png', gr)
        # shutil.move(f'{sys.argv[1]}/{name}.png', f'{sys.argv[2]}/x/batch_{batch_count}/{name}.png')
        # gr_chunk[n][:loaded_img.shape[0],:loaded_img.shape[1]] = loaded_img
    # imsave(f'{sys.argv[2]}/x/batch_{batch_count}/batch_{batch_count}.png', gr_chunk.reshape((len(batch)* 128, pad_size)))
    # np.save(f'{sys.argv[2]}/x/batch_{batch_count}/batch_{batch_count}.npy', gr_chunk)
    batch_count += 1




    