import os 

def read_dictionary(file_path):
    w = []
    with open(file_path, 'r', encoding='utf-8') as f :
        for line in f.readlines():
            w.append(line.strip())
    return w             