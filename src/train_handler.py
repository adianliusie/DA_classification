import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from .helpers import ConvHandler, Batcher, DirManager
from .utils import (no_grad, toggle_grad, make_optimizer, 
                    make_scheduler)
from .models import make_model

class TrainHandler:
    """"base class for running all sequential sentence/utterance 
        classification experiments"""
    
    def __init__(self):
        pass

    ######  Methods For Dialogue Act Classification  ##########
    
    def train(self, args:namedtuple):
        """train sequential sentence classification model. Saves 
           model in log dir and loads best epoch as self.model"""

        self.dir = DirManager(args.exp_name, 
                              args.temp)
        
        self.dir.save_args('train_args', args)
        self.mode = args.mode
        
        self.C = ConvHandler(label_path=args.label_path, 
                             system=args.system, 
                             punct=args.punct, 
                             action=args.action, 
                             hes=args.hes)

        train = self.C.prepare_data(path=args.train_path, 
                                    lim=args.lim)
        if args.dev_path:
            dev = self.C.prepare_data(path=args.dev_path, 
                                      lim=args.lim)

        self.batcher = Batcher(mode=args.mode, 
                               num_labels=args.num_labels,
                               max_len=args.max_len)
        
        self.model = make_model(system = args.system, 
                                mode = args.mode,
                                num_labels = args.num_labels, 
                                extra = self.C.tokenizer.eos_token_id)

        self.device = args.device
        self.to(self.device)

        optimizer = make_optimizer(opt_name=args.optim, 
                                   lr=args.lr, 
                                   params=self.model.parameters())
        if args.sched:
            steps = (len(train)*args.epochs)/args.bsz
            scheduler = make_scheduler(optimizer=optimizer, 
                                       steps=steps,
                                       mode=args.sched)

        best_metric = 0
        for epoch in range(args.epochs):
            logger = np.zeros(3)
            self.model.train()
            train_batches = self.batcher.batches(train, 
                                                 bsz=args.bsz, 
                                                 shuf=True)
            
            for k, batch in enumerate(train_batches, start=1):
                #forward and loss calculation
                output = self.model_output(batch)
                loss = output.loss

                #updating model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #update scheduler if step with each batch
                if args.sched == 'triangular': 
                    scheduler.step()

                #accuracy logging
                logger += [output.loss.item(), 
                           output.hits, 
                           output.num_preds]

                #print every now and then
                if k%args.print_len == 0:
                    self.dir.log(f'{epoch:<3} {k:<5} ',
                                 f'loss {logger[0]/args.print_len:.3f} ',
                                 f'acc {logger[1]/logger[2]:.3f}')
                    logger = np.zeros(3)
                            
            if args.dev_path:
                logger = np.zeros(3)
                self.model.eval()
                
                dev_batches = self.batcher.batches(dev, 
                                                   bsz=args.bsz, 
                                                   shuf=True)
                
                for k, batch in enumerate(dev_batches, start=1):
                    output = self.model_output(batch, no_grad=True)
                    logger += [output.loss.item(), output.hits, 
                               output.num_preds]
                   
                loss = logger[0]/k
                acc = logger[1]/logger[2]
                self.dir.log(f'\n DEV {epoch:<3}   loss:{loss:.3f}   ',
                             f'acc:{acc:.3f}')
      
                if acc > best_metric:
                    self.save_model()
                    best_metric = acc

            #update scheduler if step with each epoch
            if args.sched == 'step': 
                scheduler.step()
          
        if not args.dev_path: self.save_model()
        else:                 self.load_model()

    @toggle_grad
    def model_output(self, batch):
        """flexible method for dealing with different set ups. 
           Returns loss and accuracy statistics"""
        
        if self.mode == 'seq2seq':
            output = self.model(input_ids=batch.ids, 
                                attention_mask=batch.mask, 
                                labels=batch.labels)
            loss = output.loss
            hits = torch.argmax(output.logits, dim=-1) == batch.labels
            hits = torch.sum(hits[batch.labels != -100]).item()
  
        if self.mode == 'encoder':
            y = self.model(ids=batch.ids, mask=batch.mask, )
            loss = F.cross_entropy(
                y.view(-1, y.size(2)), batch.labels.view(-1))
            hits = torch.argmax(y, dim=-1) == batch.labels
            hits = torch.sum(hits[batch.labels != -100]).item()
            
        if self.mode == 'context':
            y = self.model(input_ids=batch.ids, 
                           attention_mask=batch.mask)
            loss = F.cross_entropy(y, batch.labels)

        num_preds = torch.sum(batch.labels != -100).item()
        return SimpleNamespace(loss=loss, hits=hits, 
                               num_preds=num_preds)

    #############   SAVING AND LOADING    #############
    
    def save_model(self, name='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.path}/models/{name}.pt')
        self.model.to(self.device)

    def load_model(self, name='base'):
        self.model.load_state_dict(
            torch.load(self.dir.path + f'/models/{name}.pt'))

    def to(self, device):
        if hasattr(self, 'model'):   self.model.to(device)
        if hasattr(self, 'batcher'): self.batcher.to(device)

