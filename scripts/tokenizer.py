import numpy as np
import json
import os, sys

def make_vocabulary(YSequences):
    vocabulary = set()
    for samples in YSequences:
        for element in samples:
            vocabulary.add(element)
 
    #Vocabulary created
    w2i = {symbol:idx+1 for idx,symbol in enumerate(vocabulary)}
    i2w = {idx+1:symbol for idx,symbol in enumerate(vocabulary)}
    
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'
 
    #Save the vocabulary
    np.save("w2i_aa.npy", w2i)
    np.save("i2w_aa.npy", i2w)
    with open('w2i_aa.json', 'w') as f:
        json.dump(w2i, f)
    with open('i2w_aa.json', 'w') as f:
        json.dump(i2w, f)
 
    return w2i, i2w

if __name__ == "__main__":
    path = '../MUMT 203 AMT/package_aa'
    y_sequences = []
    for idx, dirname in enumerate(os.listdir(path)):
        print(idx)
        for filename in os.listdir(f'{path}/{dirname}'):
            if filename.endswith(".semantic"):
                f = open(f'{path}/{dirname}/{filename}', 'r')
                # print(f.read().split())
                y_sequences.append(f.read().split())
    
    make_vocabulary(y_sequences)