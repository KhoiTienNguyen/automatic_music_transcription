import os, sys, shutil
import json

aa = json.load(open('no_tie_aa.json'))
ab = json.load(open('no_tie_ab.json'))

path_aa = f'/mnt/d/amt/spectrogram/aa_{sys.argv[1]}'
path_ab = f'/mnt/d/amt/spectrogram/ab_{sys.argv[1]}'
# Sys.argv[1] = In folder. Example: acoustic_guitar
# Sys.argv[2] = Out folder. Example: D:\AMT\all\acoustic_guitar
for filename in os.listdir(path_aa):
    if filename.split('.')[0] + '.wav' in aa:
        shutil.copy(f'{path_aa}/{filename}', f'{sys.argv[2]}/{filename}')
for filename in os.listdir(path_ab):
    if filename.split('.')[0] + '.wav' in ab:
        shutil.copy(f'{path_ab}/{filename}', f'{sys.argv[2]}/{filename}')
