import torch
import torch.nn as nn
from .led_patching import (LEDEncoderUttEncode, LEDModelPatched, 
                           LEDDecoderPatched, 
                           LEDModelAlpha)

class TransformerHead(torch.nn.Module):
    """wrapper where a classification head is added to trans"""
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)

    def forward(self, **kwargs):
        h   = self.transformer(**kwargs)
        h_n = h.last_hidden_state[:,0]
        y   = self.classifier(h_n)
        return y
    
    def get_embeds(self, input_ids):
        """ gets encoder embeddings for given ids"""
        embeds = self.transformer.embeddings.word_embeddings(input_ids)
        return embeds

class SequenceTransformer(torch.nn.Module):
    """transformer wrapper for sequential classification"""
    def __init__(self, transformer, num_class, sent_id=None):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)
        self.sent_id = int(sent_id)
        
    def forward(self, **kwargs):
        h = self.transformer(**kwargs)
        h_sents = self.get_sent_vectors(ids, h.last_hidden_state)
        y = self.classifier(h_sents)
        return y
    
    def get_sent_vectors(self, ids:torch.Tensor, h:torch.Tensor):
        """only selects vectors where the input id is sent_id"""
 
        #get indices of all sent vectors 
        sent_pos = (ids==self.sent_id).nonzero(as_tuple=False).tolist()
        
        #collect all vectors in each row
        output = [[] for _ in range(len(ids))]
        for conv_num, word_num in sent_pos:
            output[conv_num].append(h[conv_num,word_num].tolist())

        #pad array
        pad_elem = [0]*len(h[0,0])
        max_row_len = max([len(row) for row in output])
        padded_output = [row + [pad_elem]*(max_row_len-len(row))
                                              for row in output]

        return torch.FloatTensor(padded_output).to(h.device)

class Seq2SeqWrapper(torch.nn.Module):
    @classmethod 
    def wrap(cls, transformer, num_class):
        """encoder-decoder wrapper to change decoder embeddings dim."""

        #here the start, end and pad token have ids after all the labels
        trans = transformer
        trans.start_idx  = num_class
        trans.end_idx    = num_class+1
        trans.pad_idx    = num_class+2
        trans.num_tokens = num_class+3

        #updating config details    
        transformer.config.vocab_size = trans.num_tokens 
                    #^bad as this attr for both encoder and decoder
        transformer.config.decoder_start_token_id = trans.start_idx

        #reducing embedding matrix to new decoder vocab
        d_model = transformer.config.d_model
        transformer.model.decoder.embed_tokens = \
                 nn.Embedding(trans.num_tokens, d_model, trans.pad_idx)

        #reformatting the head 
        transformer.lm_head = nn.Linear(d_model, 
                                        trans.num_tokens, 
                                        bias=False)
        transformer.register_buffer("final_logits_bias", 
                                    torch.zeros((1, trans.num_tokens)))
        
        #add Seq2SeqWrapper as a base class
        trans_cls = transformer.__class__ 
        transformer.__class__ = type(cls.__name__, (trans_cls, cls), {})
        return transformer 

    def get_embeds(self, input_ids):
        """ gets encoder embeddings for given ids"""
        embeds = self.model.encoder.embed_tokens(input_ids)
        return embeds

    def set_setting(self, mode:str):
        if mode == None:
            print('using baseline seq2seq set up')

        if mode == 'pre_encoder':
            print('adding utterance embeddings BEFORE encoder')
            self.model.encoder.__class__ = LEDEncoderUttEncode
            decoder_pos_embedding = self.model.decoder.embed_positions
            self.model.encoder.set_up(embeddings=decoder_pos_embedding,
                                      sep_token=2)
        elif mode == 'post_encoder': 
            print('adding utterance embeddings AFTER encoder')
            self.model.__class__ = LEDModelAlpha

        elif mode == 'pre_decoder':
            print('adding word embeddings to decoder')
            self.model.__class__ = LEDModelPatched
            self.model.decoder.__class__ = LEDDecoderPatched
                
        else:
            raise ValueError('Invalid system argument')

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
