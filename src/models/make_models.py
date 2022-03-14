import torch

from .model_src import TransformerHead, Seq2SeqWrapper, SequenceTransformer
from .hugging_utils import get_transformer

def make_model(system, mode:str, num_labels:int=None, dir_obj=None, 
               eos_tok=None, system_args=None)->torch.nn.Module:
    """ creates the sequential classification model """
    if dir_obj == None: 
        dir_obj = type('redirects_log_to_print', (),{'log':print})
            
    transformer = get_transformer(system)
    dir_obj.log(f'using the following transformer mode: {system}')
    if mode == 'seq2seq':
        model = Seq2SeqWrapper.wrap(transformer, 
                                      num_labels)
        if not system_args:
            dir_obj.log('using baseline seq2seq set up')
        else:
            model.set_setting(system_args, dir_obj=dir_obj)

    if mode == 'context':
        model = TransformerHead(transformer, 
                                num_labels)
        dir_obj.log('using basic context set up')

    if mode == 'encoder':
        dir_obj.log('using transformer encoder set up')
        model = SequenceTransformer(transformer, 
                                    num_labels, 
                                    sent_id=extra)
    return model
