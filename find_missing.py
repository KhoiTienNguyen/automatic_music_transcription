import os, sys
import json

s = set()
for filename in os.listdir('/mnt/d/amt/aa_violin'):
    s.add(filename)

f = open('order.json', 'r')
data = json.load(f)

for i in data:
    if i not in s:
        print(i)