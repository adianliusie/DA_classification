import torch

from .models import TransformerHead, Seq2SeqWrapper, SequenceTransformer
from .hugging_utils import get_transformer

def make_model(system, mode:str, num_labels:int=None, 
               eos_tok=None, system_args=False)->torch.nn.Module:
    """ creates the sequential classification model """
    transformer = get_transformer(system)
    if mode == 'seq2seq':
        model = Seq2SeqWrapper.wrap(transformer, 
                                      num_labels)
        model.set_setting(system_args)
        
    if mode == 'context':
        model = TransformerHead(transformer, 
                                num_labels)
    if mode == 'encoder':
        model = SequenceTransformer(transformer, 
                                    num_labels, 
                                    sent_id=extra)
    return model
