import torch
from typing import List, Tuple
from types import SimpleNamespace
import random

from ..utils import flatten

class Batcher():
    """class that creates batches for training/eval"""
    
    def __init__(self, mode:'str', num_labels:int=None, 
                 max_len:int=None, system_args:tuple=None):
        self.mode = mode
        self.max_len = max_len
        self.device = torch.device('cpu')
        self.num_labels = num_labels
        self.system_args = system_args
        
        if mode == 'context': 
            self.past, self.fut = int(system_args[0]), int(system_args[1])
            self.conv_prep = self._prep_context
        else:                 
            self.conv_prep = self._prep_conv_seq
            
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
        ids, labels, spkr_ids, utt_ids = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids, _ = self._get_padded_ids(spkr_ids)
        utt_ids, _ = self._get_padded_ids(utt_ids)

        if self.mode in ['seq2seq', 'encoder']: 
            labels = self._pad_labels(labels)
        labels   = torch.LongTensor(labels).to(self.device)

        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                               spkr_ids=spkr_ids, utt_ids=utt_ids)
    
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
    
    def _prep_conv_seq(self, data:List['Conversations']):
        """ data preparation when input is the entire conversation"""
        output = []
        for conv in data:
            #get all utterances in conv and labels
            ids = [utt.ids for utt in conv.utts]
            ids = self.proc_utts_seq(utt_ids)

            labels = [utt.label for utt in conv.utts]
            spkrs  = [utt.spkr_id for utt in conv.utts]
            print(spkrs)
            #get utterances meta information
            spkr_ids = [[s]*len(i) for s, i in zip(spkrs, utt_ids)]
            spkr_ids = flatten(spkr_ids)
            utt_ids = [[k]*len(i) for k, i in enumerate(utt_ids)]
            utt_ids = flatten(utt_ids)
            ids = flatten(ids)
            output.append([ids, labels, spkr_ids, utt_ids])
        return output
    
    def proc_utts_seq(self, utts:List[list]):
        CLS, SEP = utts[0][0], utts[0][-1]
        if 'utt_embed' in self.system_args:
            utt_ids = [utt[1:-1] for utt in utts]
        else:
            utt_ids = [utt.ids[1:] for utt in conv.utts[1:]]
            utt_ids[0] = [CLS] + utt_ids[0]
        return utt_ids
                       
    def _prep_context(self, data:List['Conversations'], train:bool=True):
        """ data preparation for context"""
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                #get past, current and future utterances within the window
                w_s, w_e = max(i-self.past, 0), i+self.fut
                past_u = [utt.ids[1:-1] for utt in conv.utts[w_s:i]]
                cur_u  = cur_utt.ids
                fut_u  = [utt.ids[1:-1] for utt in conv.utts[i+1:w_e+1]]
                
                #get speaker ids
                spkrs  = [utt.spkr_id for utt in conv.utts[w_s:w_e+1]]
                
                #prepare the tokens to be used as a flat input
                past_u, cur_u, fut_u = self.proc_utts_context(past_u, cur_u, fut_u)
                ids = self.utt_join(past_u, cur_u, fut_u)
                
                #prepare other meta information useful for the task
                spkr_ids = [[s]*len(i) for s, i in zip(spkrs, past_u+[cur_u]+fut_u)]
                spkr_ids = flatten(spkr_ids)
                c = max(self.past-i, 0) #to ensure current utt has same utt_id
                utt_ids = [[k+c]*len(i) for k, i in enumerate(past_u+[cur_u]+fut_u)]
                utt_ids = flatten(utt_ids)
                
                ##add example to conversation examples
                conv_out.append([ids, cur_utt.label, spkr_ids, utt_ids])
            output.append(conv_out)
        if train: output = flatten(output)
        return output

    def utt_join(self, past:List[list], cur:list, fut:List[list]):
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

    def proc_utts_context(self, past:List[list], cur:list, fut:List[list]):
        """depending on mode, adds sep tokens"""
        CLS, SEP = cur[0], cur[-1]
        if 'utt_embed' in self.system_args:
            cur = cur[1:-1] #no special tokens
        elif 'utt_sep' in self.system_args:
            past = [i + [SEP] for i in past]
            fut  = [i + [SEP] for i in fut]
        elif len(self.system_args) >= 3:
            raise ValueError("invalid context mode")
        return past, cur, fut
    
    ###### General Methods for the class
    
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    def __call__(self, convH:'ConvHelper', bsz:int=4, shuf:bool=False):
        """routes the main method do the batches function"""
        return self.batches(convH, bsz, shuf)
        
                 