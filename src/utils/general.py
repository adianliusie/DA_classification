from typing import List, Tuple
from collections import namedtuple
import copy
import json

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def load_list(path:str)->list:
    with open(path, 'r') as f:
        conv_ids = f.readlines()
        conv_ids = [i.replace('\n', '') for i in conv_ids]
    return conv_ids

def flatten(x:list)->list:
    """ flattens list [[1,2,3],[4],[5,6]] to [1,2,3,4,5,6]"""
    return [tok for sent in x for tok in sent]

def pairs(x:list)->List[Tuple]: 
    """ given list [x1, x2, x3, x4], returns 
        [(x1, x2), (x2, x3), (x3, x4)]"""
    outcome = [(x[i], x[i+1]) for i in range(len(x)-1)]
    return outcome

def join_namespace(args_1:namedtuple, args_2:namedtuple):
    """given 2 namedtuple/SimpleNamespace, adds all arguents
       that aren't in the second into the first"""
    args_1 = copy.deepcopy(args_1)
    for k, v in args_2.__dict__.items():
        if k not in args_1:
            setattr(args_1, k, v)
    return args_1