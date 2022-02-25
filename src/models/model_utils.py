import torch

from .models import TransformerHead, Seq2SeqWrapper, SequenceTransformer
from .hugging_utils import get_transformer

def make_model(system, mode:str, num_labels:int=None, 
               extra=None)->torch.nn.Module:
    """ creates the sequential classification model """
    transformer = get_transformer(system)
    if mode == 'seq2seq':
        model = Seq2SeqWrapper.create(transformer, 
                                      num_labels)
        if extra == 'override':
            model.toggle_pos_encoding()
            
    if mode == 'context':
        model = TransformerHead(transformer, 
                                num_labels)
    if mode == 'encoder':
        model = SequenceTransformer(transformer, 
                                    num_labels, 
                                    sent_id=extra)
    return model
