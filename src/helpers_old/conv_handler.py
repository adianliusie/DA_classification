from types import SimpleNamespace
from typing import List
from tqdm import tqdm 

import json
import re 

from ..utils import load_json, flatten, load_list
from ..models import get_tokenizer
from ..config import config

class Utterance:
    def __init__(self, text, speaker=None, label=None, tags=None):
        self.text = text
        self.speaker = speaker
        self.label = label
        self.tags = tags
        self.ids = None
        self.label_name = None
      
    def __repr__(self):
        return self.text
    
class Conversation:
    def __init__(self, data:dict):
        self.conv_id = str(data['conv_id'])
        self.utts  = [Utterance(**utt) for utt in data['turns']] #should change to utts
        if config.debug: self.utts = self.utts[:config.debug_len]
            
        for key, value in data.items():
            if key not in ['turns', 'conv_id']:
                setattr(self, key, value)
        
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

class ConvHandler:    
    def __init__(self, label_path=None, system=None, punct=False,
                 action=False, hes=False, tqdm_disable=False):
        """ Initialises the Conversation helper """
        if system:
            self.system = system
            self.tokenizer = get_tokenizer(system)
            
        self.cleaner = TextCleaner(punct=punct, action=action, hes=hes)
        
        self.label_dict = None
        if label_path:
            label_dict = load_json(label_path)
            self.label_dict = {int(k):v for k, v in label_dict.items()}
        
        self.label_to_tok = None
        self.tqdm_disable = tqdm_disable
        
    def prepare_data(self, path:str, lim:int=None)->List[Conversation]:
        """ Given path, will load and process data for downstream tasks """
        # if json, convert data to Conversation object used by system
        if path.split('.')[-1] == 'json':
            raw_data = load_json(path)
            data = [Conversation(conv) for conv in raw_data]
        
        self.clean_text(data)
        self.get_speaker_ids(data)
        if lim: data = data[:lim]
        if self.system:     self.tok_convs(data)  
        if self.label_dict: self.get_label_names(data)
        return data
    
    def clean_text(self, data:List[Conversation]):
        """ processes text depending on arguments. E.g. punct=True filters
        punctuation, action=True filters actions etc."""
        for conv in data:
            for utt in conv:
                utt.text = self.cleaner.clean_text(utt.text)
    
    def tok_convs(self, data:List[Conversation]):
        """ generates tokenized ids for each utterance in Conversation """
        for conv in tqdm(data, disable=self.tqdm_disable):
            for utt in conv:
                utt.ids = self.tokenizer(utt.text).input_ids
    
    def get_label_names(self, data:List[Conversation]):
        """ generates detailed label name for each utterance """
        for conv in data:
            for utt in conv:
                utt.label_name = self.label_dict[utt.label]
                
    def get_speaker_ids(self, data:List[Conversation]):
        speakers = set([utt.speaker for conv in data for utt in conv])
        self.speaker_id_dict = {s:k for k, s in enumerate(speakers)}
        assert len(speakers)==2, "speaker codes have gone wrong"
        for conv in data:
            for utt in conv:
                utt.spkr_id = self.speaker_id_dict[utt.speaker]
                        
    def __getitem__(self, x:str):
        """ returns conv with a given conv_id if exists in dataset """
        for conv in self.data:
            if conv.conv_id == str(x): return conv
        raise ValueError('conversation not found')
             
    def __contains__(self, x:str):
        """ checks if conv_id exists in dataset """
        output = False
        if x in [conv.conv_id for conv in self.data]:
            output = True
        return output
       

class TextCleaner:
    def __init__(self, punct=False, action=False, hes=False):
        self.punct = punct
        self.action = action
        self.hes = hes
                 
    def clean_text(self, text:str)->str:
        """method which cleans text with chosen convention"""
        
        if self.action:
            text = re.sub("[\[\(\<\%].*?[\]\)\>\%]", "", text)    
        if self.punct: 
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
        if self.hes:
            text = self.hesitation(text)
        text = ' '.join(text.split())
        return text

    @staticmethod
    def hesitation(text:str)->str:
        """internal function to converts hesitation"""
        hes_maps = {"umhum":"um", "uh-huh":"um", 
                    "uhhuh":"um", "hum":"um", "uh":'um'}

        for h1, h2 in hes_maps.items():
            if h1 in text:
                pattern = r'(^|[^a-zA-z])'+h1+r'($|[^a-zA-Z])'
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
                #run line twice as uh uh share middle character
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
        return text 
