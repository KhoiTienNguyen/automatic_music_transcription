import numpy as np
import json
import os, sys

def make_vocabulary(vocabulary):
    # vocabulary = set()
    # for samples in YSequences:
    #     for element in samples:
    #         vocabulary.add(element)
 
    #Vocabulary created
    w2i = {symbol:idx+1 for idx,symbol in enumerate(vocabulary)}
    i2w = {idx+1:symbol for idx,symbol in enumerate(vocabulary)}
    
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'
 
    #Save the vocabulary
    # np.save("w2i_aav3.npy", w2i)
    # np.save("i2w_aav3.npy", i2w)
    with open('w2i_abv3.json', 'w') as f:
        json.dump(w2i, f)
    with open('i2w_abv3.json', 'w') as f:
        json.dump(i2w, f)
 
    return w2i, i2w

if __name__ == "__main__":
    remove_list = ['barline', 'keySignature', 'clef', 'timeSignature']
    json_path = 'no_tie_ab.json'
    f = open(json_path)
    order = json.load(f)
    path = '../MUMT 203 AMT/package_ab'
    y_sequences = set()
    count = 1
    for idx, dirname in enumerate(os.listdir(path)):
        if dirname + '.wav' not in order:
            continue
        print(count)
        count += 1
        for filename in os.listdir(f'{path}/{dirname}'):
            if filename.endswith(".semantic"):
                f = open(f'{path}/{dirname}/{filename}', 'r')
                # print(f.read().split())
                for note in f.read().split():
                    if 'barline' in note or 'keySignature' in note or 'clef' in note or 'timeSignature' in note:
                        continue
                    y_sequences.add(note)
    
    make_vocabulary(y_sequences)