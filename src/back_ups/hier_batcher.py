import torch
from typing import List, Tuple
from types import SimpleNamespace
import random
from abc import abstractmethod, ABCMeta

from ..utils import flatten

class HierBatcher(metaclass=ABCMeta):
    """base batching helper class to be inherited"""

    def __init__(self, max_len:int=None):
        self.max_len = max_len
        self.device = torch.device('cpu')
    
    def to(self, device:torch.device):
        self.device = device
        
    def conv_batches(self, convH:'ConvHelper', shuf:bool=False):
        """batches an entire conversation, and provides conv id"""
        convs = self.prep_whole(convH)
        
        return [self.batchify(conv) for conv in convs]

    def prep_whole(self, ConvH:'ConvHelper'): 
        output = []
        for conv in ConvH.data:
            utt_ids  = [utt.ids[1:] for utt in conv]
            conv_ids = [conv.utts[0].ids[0]] + flatten(utt_ids)
            labels   = [utt.label for utt in conv]
            output.append([conv_ids, labels])
        return output    
    
    def batchify(self, batch):
        ids, labels = batch
        ids = torch.LongTensor(ids).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, labels=labels)
   
                 
                 
                 
                 
                 
                 