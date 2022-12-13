import os, sys
from mel import convert
import json
from multiprocessing import Pool

if len(sys.argv) != 3:
    raise Exception("Input directory path and output directory path")

file_list = os.listdir(sys.argv[1])

def batch_to_mel(filename):
    width = convert(f'{sys.argv[1]}/{filename}', sys.argv[2])
    return (filename, width)

with Pool(2) as p:
    valid_list = p.map(batch_to_mel, file_list)
order = {k:v for (k,v) in valid_list}
with open('order_ab.json', 'w') as f:
    json.dump(order, f)