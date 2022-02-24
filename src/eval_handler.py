import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

import time #temp

from src.helpers import ConvHandler, Batcher, DirManager
from src.models import make_model
from src.utils import join_namespace, no_grad, Levenshtein
from .train_handler import TrainHandler

class EvalHandler(TrainHandler):
    """"base class for running all sequential sentence 
        evaluation and analysis on trained models"""
    
    def __init__(self, exp_name:str):
        self.dir = DirManager.load_dir(exp_name)

    ######  Methods For Dialogue Act Classification  ##########
    
    @no_grad
    def evaluate(self, args:namedtuple):
        """ evaluating model performance with loss and accurancy"""
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare data
        eval_data = self.C.prepare_data(path=args.test_path)
        eval_batches = self.batcher(eval_data, 
                                    bsz=args.bsz, 
                                    shuf=False)
        logger = np.zeros(3)
        for k, batch in enumerate(eval_batches, start=1):
            output = self.model_output(batch)
            logger += [output.loss.item(), 
                       output.hits, 
                       output.num_preds]
        
        print(f'loss {logger[0]/k:.3f}  ',
              f'acc {logger[1]/logger[2]:.3f}')
        
    @no_grad
    def evaluate_free(self, args:namedtuple):
        """ evaluating model in free running set up
            performance is assessed using Lev Dist."""
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare data
        eval_data = self.C.prepare_data(path=args.test_path, 
                                        lim=args.lim)
        eval_convs = self.batcher.eval_batches(eval_data)
        
        logger = np.zeros(5)
        for conv in tqdm(eval_convs):
            label_seq  = conv.labels.tolist()
            pred_seq   = self.model_free(conv).tolist()
            
            #if batch has to be squeezed
            if len(label_seq) == 1: 
                pred_seq  = pred_seq[0]
                label_seq = label_seq[0]

            logger[:4] += Levenshtein.wer(pred_seq, label_seq)
            logger[4]  += len(label_seq)
            
        print(f"WER:{logger[0]/logger[4]:.3f}  ",
              f"replace:{logger[1]/logger[4]:.3f}  ",
              f"inserts: {logger[2]/logger[4]:.3f}  ",
              f"deletion: {logger[3]/logger[4]:.3f}")
        
    def model_free(self, batch):
        if self.mode == 'seq2seq':
            pred = self.model.generate(
                    input_ids=batch.ids, 
                    attention_mask=batch.mask, 
                    num_beams=5,
                    bos_token_id=self.decoder_start,
                    eos_token_id=self.decoder_end,
                    pad_token_id=self.decoder_pad,
                    max_length=500
                   )
            
        if self.mode == 'context':
            y = self.model(input_ids=batch.ids, 
                           attention_mask=batch.mask)
            pred = torch.argmax(y, -1)
        
        return pred
    
    def saliency(self, args:namedtuple, N:int=50, 
                 conv_num:int=0, k:int=0):
        """ generate saliency maps for predictions """
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare conversation in interest
        conv = self.C.prepare_data(path=args.test_path)[conv_num]
        conv = self.batcher([conv], bsz=1, shuf=False)
        if self.mode == 'context': conv = conv[k]
        words = [self.C.tokenizer.decode(i) for i in conv.ids[0]]
        
        #see prediction details
        y = self.model_output(conv).logits[0]
        pred_idx = torch.argmax(y).item()      
        prob = F.softmax(y, dim=-1)[pred_idx].item()
        print(pred_idx, round(prob, 3))
        
        #using integrated gradients, where we approximate the path 
        #integral from baselnie to input using riemann sum
        with torch.no_grad():
            input_embeds = self.model.get_embeddings(conv.ids)
            base_embeds = torch.zeros_like(input_embeds)
            vec_dir = (input_embeds-base_embeds)

            alphas = torch.arange(1, N+1, device=self.device)/N
            line_path = base_embeds + alphas.view(N,1,1)*vec_dir            
            embed_batches = [line_path[i:i+args.bsz] for i in 
                             range(0, len(line_path), args.bsz)]

        #Computing the line integral, 
        output = torch.zeros_like(input_embeds)
        for embed_batch in tqdm(embed_batches):
            embed_batch.requires_grad_(True)
            y = self.model(inputs_embeds=embed_batch)
            preds = F.softmax(y, dim=-1)[:,pred_idx]
            torch.sum(preds).backward()
            
            grads = torch.sum(embed_batch.grad, dim=0)
            output += grads.detach().clone()

        #get attribution summed for each word
        tok_attr = torch.sum((output*vec_dir).squeeze(0), dim=-1)/N
        tok_attr = tok_attr.tolist()
        return words, tok_attr
    
    def set_up(self, args):
        #load training arguments and adding to args
        t_args = self.dir.load_args('train_args')
        args = join_namespace(args, t_args)
        self.mode = args.mode
        
        #load final model
        self.model = make_model(system=args.system, 
                                mode=args.mode,
                                num_labels=args.num_labels)
        self.load_model()
        self.model.eval()

        #conversation processing
        self.C = ConvHandler(label_path=args.label_path, 
                             system=args.system, 
                             punct=args.punct, 
                             action=args.action, 
                             hes=args.hes)
        
        self.batcher = Batcher(mode=args.mode, 
                               num_labels=args.num_labels,
                               max_len=args.max_len, 
                               mode_args=args.mode_args)

        #get start, pad and end token for decoder
        if self.mode == 'seq2seq':
            self.decoder_start = args.num_labels
            self.decoder_end   = args.num_labels+1
            self.decoder_pad   = args.num_labels+2
            
        #set to device
        self.device = args.device
        self.to(self.device)

        return args
    

        