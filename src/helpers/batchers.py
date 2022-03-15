import torch
from typing import List, Tuple
from types import SimpleNamespace
import random

from ..utils import flatten
from abc import ABCMeta, abstractmethod


def make_batcher(mode:str, max_len:int=None, 
                 formatting:str=None, system_args:tuple=None):
    if mode == 'context':
        batcher = ContextBatcher(max_len, formatting, system_args) 
    elif mode in ['seq2seq' or 'seq_encoding']:
        batcher = SeqBatcher(max_len, formatting)
    return batcher


class BaseBatcher:
    """base class that creates batches for training/eval for all tasks"""

    def __init__(self, max_len:int=None, formatting:str=None):
        """initialises object"""
        self.device = torch.device('cpu')
        self.max_len = max_len
        self.formatting = formatting
    
    @abstractmethod
    def seq_cls_batches(self): pass

    @abstractmethod
    def seq_cls_eval_batches(self): pass
    
    def _get_padded_ids(self, ids:list)->("padded ids", "padded_mask"):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def _pad_seq(self, x:list, pad_val:int=0)->list:
        """pads input so can be put in a tensor"""
        max_len = max([len(i) for i in x])
        x_pad = [i + [pad_val]*(max_len-len(i)) for i in x]
        x_pad = torch.LongTensor(x_pad).to(self.device)
        return x_pad
       
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    def __call__(self, convH:'ConvHelper', bsz:int=4, shuf:bool=False):
        """routes the main method do the batches function"""
        return self.batches(convH, bsz, shuf)
    
    
class ContextBatcher(BaseBatcher):
    def __init__(self, max_len:int=None, formatting:str=None, 
                                         system_args:tuple=None):
        super().__init__(max_len=max_len, formatting=formatting)
        self.past, self.fut = int(system_args[0]), int(system_args[1])
        
    def seq_cls_batches(self, data:List['Conversations'], 
                              bsz:int, shuf:bool=False):
        convs = self._prep_utts(data, return_conv=False)
        if shuf: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        batches = [self._seq_cls_batchify(batch) for batch in batches] 
        if self.max_len:
            assert max([len(b.ids[0]) for b in batches]) <= self.max_len       
        return batches        
    
    def _seq_cls_batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkr_ids, utt_ids, utts = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids = self._pad_seq(spkr_ids)
        utt_ids = self._pad_seq(utt_ids)
        
        labels = [utt.label for utt in utts] 
        labels = torch.LongTensor(labels).to(self.device)
        #^keep in mind labels are wrong for seq2seq training
        
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                               spkr_ids=spkr_ids, utt_ids=utt_ids)
    
    def _prep_utts(self, data:List['Conversations'], return_conv:bool=False):
        """ data preparation for context, note each example is an utterance"""
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                #get past, current and future utterances within the window
                w_s, w_e = max(i-self.past, 0), i+self.fut
                cur_u  = cur_utt.ids
                past_u = [utt.ids[1:-1] for utt in conv.utts[w_s:i]]
                fut_u  = [utt.ids[1:-1] for utt in conv.utts[i+1:w_e+1]]

                #get speaker ids
                spkrs  = [utt.spkr_id for utt in conv.utts[w_s:w_e+1]]

                #prepare the tokens to be used as a flat input
                past_u, cur_u, fut_u = self._format_ids(past_u, cur_u, fut_u, spkrs)
                ids = flatten(past_u) + cur_u + flatten(fut_u)

                #prepare other meta information useful for the task
                spkr_ids = [[s]*len(i) for s, i in zip(spkrs, past_u+[cur_u]+fut_u)]
                spkr_ids = flatten(spkr_ids)
                c = max(self.past-i, 0) #to ensure current utt has same utt_id
                utt_ids = [[k+c]*len(i) for k, i in enumerate(past_u+[cur_u]+fut_u)]
                utt_ids = flatten(utt_ids)

                ##add example to conversation examples
                conv_out.append([ids, spkr_ids, utt_ids, cur_utt])
            output.append(conv_out)

        return output if return_conv else flatten(output)
        
    def _format_ids(self, past:List[list], cur:list, fut:List[list], spkrs):
        """depending on mode, adds sep tokens"""
        CLS, SEP = cur[0], cur[-1]
        if self.formatting == None:
            cur = [CLS] + cur[1:-1] + [SEP] #line does nothing, makes format clear
        elif self.formatting == 'no_special':
            cur = cur[1:-1] #no special tokens
        elif self.formatting == 'utt_sep':
            past = [i + [SEP] for i in past]
            fut  = [i + [SEP] for i in fut]
        elif self.formatting == 'spkr_sep':
            raise ValueError('this is complicated must think through more? hmmm')
        else:
            raise ValueError('invalid context formatting')
        return past, cur, fut

    def seq_cls_conv_batches(self, data:List['Conversations'], 
                                   bsz:int, shuf:bool=False):
        convs = self._prep_utts(data, return_conv=True)
        if shuf: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        batches = [self._seq_cls_batchify(batch) for batch in batches] 
        if self.max_len:
            assert max([len(b.ids[0]) for b in batches]) <= self.max_len       
        return batches   
    
    def seq_cls_eval_batches(self, data, bsz):
        conv_batches = self.seq_cls_conv_batches(data, bsz=1, shuf=False)
        return conv_batches
    
    
class SeqBatcher(BaseBatcher):
    def seq_cls_batches(self, data:List['Conversations'], 
                        bsz:int, shuf:bool=False):
        """batches an entire conversation, and provides conv id"""
        convs = self._prep_convs(data)
        if shuf: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        batches = [self._seq_cls_batchify(batch) for batch in batches] 
        if self.max_len:
            assert max([len(b.ids[0]) for b in batches]) <= self.max_len       
        return batches
    
    def _seq_cls_batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkr_ids, utt_ids, convs = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids = self._pad_seq(spkr_ids)
        utt_ids = self._pad_seq(utt_ids)
        
        labels = [[utt.label for utt in conv] for conv in convs]
        labels = self._pad_seq(labels, pad_val=-100)
        #^keep in mind labels are wrong for seq2seq training
        
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                               spkr_ids=spkr_ids, utt_ids=utt_ids)
    
    def _prep_convs(self, data:List['Conversations']):
        """ sequence classification input data preparation"""
        output = []
        for conv in data:
            #get all utterances in conv and labels
            ids = [utt.ids for utt in conv.utts]
            spkrs = [utt.spkr_id for utt in conv.utts]
            ids = self._format_ids(ids, spkrs)

            #get utterance meta information
            spkr_ids = [[s]*len(i) for s, i in zip(spkrs, ids)]
            spkr_ids = flatten(spkr_ids)
            utt_ids = [[k]*len(i) for k, i in enumerate(ids)]
            utt_ids = flatten(utt_ids)
            ids = flatten(ids)
            output.append([ids, spkr_ids, utt_ids, conv])
        return output
    
    def _format_ids(self, utts, spkrs):
        CLS, SEP = utts[0][0], utts[0][-1]
        if self.formatting == None:
            utt_ids = [utt[1:-1] for utt in utts]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_ids[-1] = utt_ids[-1] + [SEP]
        elif self.formatting == 'utt_sep':
            utt_ids = [utt.ids[1:] for utt in utts]
            utt_ids[0] = [CLS] + utt_ids[0]
        elif self.formatting == 'spkr_sep':
            print('this functionality has not been tested yet!')
            assert len(utts) == len(spkrs), "something went wrong with spkr_sep"
            speaker_tokens = [f'[SPKR_{i}]' for i in spkrs]
            utt_ids = [[s] + utt.ids[1:-1] for utt, s in zip(utts, spkrs)]
        else:
            raise ValueError('invalid sequence formatting')
        return utt_ids
        
    def seq_cls_eval_batches(self, data, bsz):
        conv_batches = self.seq_cls_batches(data, bsz=1, shuf=False)
        return conv_batches
        
    
    
    
    
    
    
    