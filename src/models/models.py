import torch
import torch.nn as nn

class TransformerHead(torch.nn.Module):
    """wrapper of transformer, where a classification head is added"""
    def __init__(self, transformer, class_num):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, class_num)

    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state[:,0]
        y = self.classifier(h1)
        return y

def Seq2SeqWrapper(transformer, num_class):
    """wrapper of encoder-decoder model to have different decoder embeddings.
       here the start, end and pad token have ids after all the labels"""
    start_idx  = num_class
    end_idx    = num_class+1
    pad_idx    = num_class+2
    num_tokens = num_class+3
    
    #updating config details    
    transformer.config.vocab_size = num_tokens
    transformer.config.decoder_start_token_id = start_idx
    transformer.config. y = pad_idx
    
    #changing embedding matrix to be over decoder vocab instead of all words
    d_model = transformer.config.d_model
    transformer.model.decoder.embed_tokens = nn.Embedding(num_tokens, d_model, pad_idx)
    
    #reformatting the head 
    transformer.lm_head = nn.Linear(d_model, num_class+3, bias=False)
    transformer.register_buffer("final_logits_bias", torch.zeros((1, num_tokens)))

    return transformer