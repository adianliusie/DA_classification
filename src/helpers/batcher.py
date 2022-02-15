import torch
from typing import List, Tuple
from types import SimpleNamespace
import random

from ..utils import flatten
from ..config import config

class Batcher():
    """base batching helper class to be inherited"""

    def __init__(self, mode:'str', max_len:int=None):
        self.mode = mode
        self.max_len = max_len
        self.device = torch.device('cpu')

    def batches(self, convH:'ConvHelper', bsz:int=2, shuf:bool=False):
        """batches an entire conversation, and provides conv id"""
        convs = self._prep_convs(convH)
        if shuf: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        batches = [self._batchify(batch) for batch in batches] 
        if self.max_len:
            assert max([len(batch.ids[0]) for batch in batches]) <= self.max_len       
        return batches
    
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    ## Methods for prepping conversations to each prediction step
    
    def _prep_convs(self, data:List['Conversations'], label_dict:dict=None): 
        """Goes through Conv Object and creates ids and label sequence"""
        if self.mode in ['seq2seq', 'encoder']:
            output = self._prep_seq2seq(data)
        elif self.mode == 'context':
            output = self._prep_context(data)
        return output
 
    def _prep_seq2seq(self, data:List['Conversations']):
        """ data preparation when input is the entire conversation"""
        output = []
        for conv in data:
            if config.debug:
                conv.utts = conv.utts[:random.randint(35,40)]
            utt_ids_1 = [utt.ids[1:] for utt in conv.utts[1:]]
            conv_ids  = conv.utts[0].ids + flatten(utt_ids_1)
            labels    = [utt.label for utt in conv.utts]
            output.append([conv_ids, labels])
        return output

    def _prep_context(self, data:List['Conversations'], train:bool=False):
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [utt.ids[1:-1] for utt in conv.utts[max(i-self.past, 0):i]]
                future_utts = [utt.ids[1:-1] for utt in conv.utts[i+1:i+self.fut+1]]
                ids = self._utt_join(past_utts, cur_utt.ids, future_utts)
                conv_out.append([ids, cur_utt.label])
            output.append(conv_out)
        if train: output = [utt for conv in output for utt in conv]
        return output

    def _utt_join(self, past:list, cur:List[list], fut:List[list])->list:
        """ given current turn, past turns and future turns, combines them """
        if self.max_u and max(len(past), len(fut)) != 0:
            k = 0
            while len(flatten(past)+cur+flatten(fut)) > self.max_u and len(past)>0:
                if k%2 == 0: past = past[1:]
                else:        fut  = fut[:-1]
                k += 1
        return flatten(past) + cur + flatten(fut)
        
    def _make_labels(self, labels:list, label_dict:dict):
        return [label_dict[i] for i in labels]
 
    ## Methods for making the batches

    def _batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, labels = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        if self.mode == 'seq2seq':   labels = self._pad_labels(labels)
        elif self.mode == 'encoder': labels = self._mask_labels(labels, ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)
    
    def _pad_labels(self, labels:list):
        """pads labels for batch so that all sequences are of the same len"""
        max_len = max([len(x) for x in labels])
        padded_labels = [x + [-100]*(max_len-len(x)) for x in labels]
        return padded_labels
        
    def _get_padded_ids(self, ids:list):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

                     
                 
                 
                 
                 
                 
