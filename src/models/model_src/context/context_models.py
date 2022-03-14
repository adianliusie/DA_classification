import torch
import torch.nn as nn

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

    