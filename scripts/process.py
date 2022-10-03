import os, sys
from mel import convert
import json

if len(sys.argv) != 3:
    raise Exception("Input directory path and output directory path")

order = {}

for filename in os.listdir(sys.argv[1]):
    # if filename.endswith(".wav"):
    width = convert(f'{sys.argv[1]}/{filename}', sys.argv[2])
    order[filename] = width

with open('test_order.json', 'w') as f:
    json.dump(order, f)