import torch
from typing import List, Tuple
from types import SimpleNamespace
import random

from ..utils import flatten

class Batcher():
    """class that creates batches for training/eval"""
    
    def __init__(self, mode:'str', num_labels:int=None, 
                 max_len:int=None, mode_args:tuple=None):
        self.mode = mode
        self.max_len = max_len
        self.device = torch.device('cpu')
        self.num_labels = num_labels
        
        if mode == 'context': 
            self.past, self.fut = mode_args
            self.conv_prep = self._prep_context
        else:                 
            self.conv_prep = self._prep_conv_whole
            
    def batches(self, data:List['Conversations'], bsz:int=4, shuf:bool=False):
        """batches an entire conversation, and provides conv id"""
        convs = self.conv_prep(data)
        if shuf: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        batches = [self._batchify(batch) for batch in batches] 
        if self.max_len:
            assert max([len(b.ids[0]) for b in batches]) <= self.max_len       
        return batches

    def eval_batches(self, data:List['Conversations']):
        if self.mode == 'context':
            convs = self._prep_context(data, train=False)
            return [self._batchify(conv) for conv in convs] 
        else:
            return self.batches(data, 1, shuf=False)
        
    ### batch preparing methods

    def _batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, labels = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        
        if self.mode in ['seq2seq', 'encoder']: 
            labels = self._pad_labels(labels)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)
    
    def _get_padded_ids(self, ids:list):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def _pad_labels(self, labels:list):
        """pads labels for batch so that all sequences are of the same len"""
        end_tok = self.num_labels + 1
        max_len = max([len(x) for x in labels])
        padded_labels = [x+[end_tok]+[-100]*(max_len-len(x)) for x in labels]
        return padded_labels
    
    ### Conv prepping methods
    
    def _prep_conv_whole(self, data:List['Conversations']):
        """ data preparation when input is the entire conversation"""
        output = []
        for conv in data:
            utt_ids_1 = [utt.ids[1:] for utt in conv.utts[1:]]
            conv_ids  = conv.utts[0].ids + flatten(utt_ids_1)
            labels    = [utt.label for utt in conv.utts]
            output.append([conv_ids, labels])
        return output
    
    def _prep_context(self, data:List['Conversations'], train:bool=True):
        """ data preparation for context"""
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [utt.ids[1:-1] for utt in conv.utts[max(i-self.past, 0):i]]
                future_utts = [utt.ids[1:-1] for utt in conv.utts[i+1:i+self.fut+1]]
                ids = self.utt_join(past_utts, cur_utt.ids, future_utts)
                conv_out.append([ids, cur_utt.label])
            output.append(conv_out)
        if train: output = flatten(output)
        return output

    def utt_join(self, past, cur, fut):
        """given past utterances, current utterance and future utterances,
           return the joined sequence under max len"""
        if self.max_len and max(len(past), len(fut)) != 0:
            k = 0
            while len(flatten(past)+cur+flatten(fut)) > self.max_len and len(past)>0:
                if k%2 == 0: past = past[1:]
                else:        fut  = fut[:-1]
                k += 1
        output = flatten(past) + cur + flatten(fut)
        return output    
 
    ###### General Methods for the class
    
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    def __call__(self, convH:'ConvHelper', bsz:int=4, shuf:bool=False):
        """routes the main method do the batches function"""
        return self.batches(convH, bsz, shuf)
        
                 
                 
                 
