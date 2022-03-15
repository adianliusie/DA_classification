import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from .helpers import ConvHandler, make_batcher, DirManager
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
        self.system_args = args.system_args
        
        self.C = ConvHandler(system=args.system, 
                             punct=args.punct, 
                             action=args.action, 
                             hes=args.hes)

        train = self.C.prepare_data(path=args.train_path, 
                                    lim=args.lim)
        if args.dev_path:
            dev = self.C.prepare_data(path=args.dev_path, 
                                      lim=args.lim)

        self.batcher = make_batcher(mode=args.mode, 
                                    max_len=args.max_len, 
                                    formatting=args.formatting,
                                    system_args=args.system_args)
        
        self.model = make_model(system = args.system, 
                                mode = args.mode,
                                num_labels = args.num_labels, 
                                dir_obj=self.dir,
                                eos_tok = self.C.tokenizer.eos_token_id,
                                system_args = args.system_args)

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

        best_epoch = (-1, 10000, 0)
        for epoch in range(args.epochs):
            logger = np.zeros(3)
            self.model.train()
            train_batches = self.batcher.seq_cls_batches(
                                train, bsz=args.bsz, shuf=True)
            
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
                    loss = f'{logger[0]/args.print_len:.3f}'
                    acc  = f'{logger[1]/logger[2]:.3f}'
                    self.dir.update_curve('train', epoch, loss, acc)
                    self.dir.log(f'{epoch:<3} {k:<5}  ',
                                 f'loss {loss}   acc {acc}')
                    logger = np.zeros(3)
                            
            if args.dev_path:
                logger = np.zeros(3)
                self.model.eval()
                
                dev_batches = self.batcher.seq_cls_batches(
                                dev, bsz=args.bsz, shuf=False)
                
                for k, batch in enumerate(dev_batches, start=1):
                    output = self.model_output(batch, no_grad=True)
                    logger += [output.loss.item(), output.hits, 
                               output.num_preds]
                   
                loss = logger[0]/k
                acc = logger[1]/logger[2]
                self.dir.log(f'{epoch:<3} DEV     LOSS:{loss:.3f}  ',
                             f'ACC:{acc:.3f} \n')
                self.dir.update_curve('dev', epoch, loss, acc)

                if acc > best_epoch[2]:
                    self.save_model()
                    best_epoch = (epoch, loss, acc)

                ##Testing
                if epoch == best_epoch[0]+5:
                    break
                
            #update scheduler if step with each epoch
            if args.sched == 'step': 
                scheduler.step()
          
        if not args.dev_path: self.save_model()
        else:                 self.load_model()
        
        self.dir.log(f'epoch {best_epoch[0]}  ',
                     f'loss: {best_epoch[1]:.3f}  ',
                     f'acc: {best_epoch[2]:.3f}')
        
    @toggle_grad
    def model_output(self, batch):
        """flexible method for dealing with different set ups. 
           Returns loss and accuracy statistics"""
        
        model_inputs = {'input_ids':batch.ids, 
                        'attention_mask':batch.mask}
        if 'spkr_embed' in self.system_args:
            model_inputs['token_type_ids'] = batch.spkr_ids
        if 'utt_embed' in self.system_args:
            model_inputs['utt_embed_ids'] = batch.utt_ids
        if self.mode == 'seq2seq':
            model_inputs['labels'] = batch.labels
            
        if self.mode == 'context':
            y = self.model(**model_inputs)           
            loss = F.cross_entropy(y, batch.labels)

        elif self.mode == 'encoder':
            y = self.model(**model_inputs)
            loss = F.cross_entropy(
                        y.view(-1, y.size(2)), batch.labels.view(-1))

        elif self.mode == 'seq2seq':
            output = self.model(**model_inputs)
            loss = output.loss
            y    = output.logits

        hits = torch.argmax(y, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()
        return SimpleNamespace(loss=loss, 
                               logits=y,
                               hits=hits, 
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

