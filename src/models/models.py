import torch
import torch.nn as nn

class SequenceTransformer(torch.nn.Module):
    """transformer wrapper for sequential classification"""
    def __init__(self, transformer, num_class, sent_id=None):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)
        self.sent_id = int(sent_id)
        
    def forward(self, ids, mask):
        h = self.transformer(input_ids=ids, 
                             attention_mask=mask).last_hidden_state
        h_sents = self.get_sent_vectors(ids, h)
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

class TransformerHead(torch.nn.Module):
    """wrapper where a classification head is added to trans"""
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)

    def forward(self, ids, mask):
        h   = self.transformer(input_ids=ids, attention_mask=mask)
        h_n = h1.last_hidden_state[:,0]
        y   = self.classifier(h_n)
        return y

def Seq2SeqWrapper(transformer, num_class):
    """encoder-decoder wrapper to change decoder embeddings dim."""
    
    #here the start, end and pad token have ids after all the labels
    start_idx  = num_class
    end_idx    = num_class+1
    pad_idx    = num_class+2
    num_tokens = num_class+3
    
    #updating config details    
    transformer.config.vocab_size = num_tokens #careful as this attr for both encoder and decoder
    transformer.config.decoder_start_token_id = start_idx
    #transformer.config.pad_token_id = pad_idx
    
    #reducing embedding matrix to new decoder vocab
    d_model = transformer.config.d_model
    transformer.model.decoder.embed_tokens = \
                           nn.Embedding(num_tokens, d_model, pad_idx)
    
    #reformatting the head 
    transformer.lm_head = nn.Linear(d_model, num_class+3, bias=False)
    transformer.register_buffer("final_logits_bias", 
                                torch.zeros((1, num_tokens)))

    return transformer