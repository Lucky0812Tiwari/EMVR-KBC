import os
import json

def is_empty_dir(directory):
    return not bool(os.listdir(directory))

def file_exists(file_path):
    if os.path.exists(file_path):
     return True
    abs_path = os.path.join(os.getcwd(), file_path)
    if os.path.exists(abs_path):
        return True
    return False

def save_nested_list(nested_list, fname):
    with open(fname, 'w') as file:
        for sublist in nested_list:
            json.dump(sublist, file)
            file.write('\n')

def load_nested_list(fname):
    nested_list = []
    with open(fname, 'r') as file:
        for line in file:
            sublist = json.loads(line.strip())
            nested_list.append(sublist)
    return nested_list

def _build_dict_from_lst(l):
    d = dict()
    for idx in range(len(l)):
        d[idx]=l[idx]
    return d

def _build_inverse_dict(d):
    assert len(d.items()) == len(list(set(d.items())))
    inv_d = {v: k for k, v in d.items()}
    return inv_d

def _build_dict_from_file(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(maxsplit=1)
            result_dict[int(key)] = str(value)
    return result_dict
