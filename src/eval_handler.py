import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from src.helpers import ConvHandler, Batcher, DirManager
from src.models import make_model
from src.utils import join_namespace, no_grad, Levenshtein
from .train_handler import TrainHandler

class EvalHandler(TrainHandler):
    """"base class for running all sequential sentence 
        evaluation on trained models"""
    
    def __init__(self, exp_name):
        self.dir = DirManager.load_dir(exp_name)

    ######  Methods For Dialogue Act Classification  ##########
    
    @no_grad
    def evaluate(self, args:namedtuple):
        """ evaluating model performance with loss and accurancy"""
        
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
        
        eval_data = self.C.prepare_data(path=args.test_path)
        self.batcher = Batcher(args.mode)

        #set to device and init logging
        self.device = args.device
        self.to(self.device)
        logger = np.zeros(3)

        eval_batches = self.batcher(eval_data, 
                                    bsz=args.bsz, 
                                    shuf=False)
        
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

        #get start, pad and end token for decoder
        self.decoder_start = args.num_labels
        self.decoder_end   = args.num_labels+1
        self.decoder_pad   = args.num_labels+2

        #conversation processing
        self.C = ConvHandler(label_path=args.label_path, 
                             system=args.system, 
                             punct=args.punct, 
                             action=args.action, 
                             hes=args.hes)
        
        eval_data = self.C.prepare_data(path=args.test_path)
        self.batcher = Batcher(args.mode)

        #set to device and init logging
        self.device = args.device
        self.to(self.device)
        logger = np.zeros(3)

        eval_batches = self.batcher(eval_data, 
                                    bsz=args.bsz, 
                                    shuf=False)
        
        for k, batch in enumerate(eval_batches, start=1):
            self.model_free(batch)
            
    def model_free(self, batch):     
        pred = self.model.generate(
                   input_ids=batch.ids, 
                   attention_mask=batch.mask, 
                   num_beams=5, 
                   bos_token_id=self.decoder_start,
                   eos_token_id=self.decoder_end,
                   pad_token_id=self.decoder_pad,
                   max_length=200
                )
        
        print(pred)
        print(batch.labels)
