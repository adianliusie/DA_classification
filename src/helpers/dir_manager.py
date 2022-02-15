import os
import json
import torch
import shutil

from types import SimpleNamespace
from typing import Callable
from collections import namedtuple

from ..utils import load_json
from ..config import config


BASE_DIR = f'{config.base_dir}/experiments/model_files'

class DirManager:
    """ File managing class which saves logs, models and config files """
    def __init__(self, exp_name:str=None, temp:bool=False):
        if temp:
            print("using temp directory")
            self.exp_name = 'temp'
            self.del_temp_dir()
        else:
            self.exp_name = exp_name

        self.make_dir()
        self.log = self.make_logger(file_name='log')

    def del_temp_dir(self):
        if os.path.isdir(f'{BASE_DIR}/temp'): shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def save_args(self, name:str, args:namedtuple):
        config_path = f'{self.path}/{name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(args.__dict__, jsonFile, indent=4)

    def make_logger(self, file_name:str)->Callable:     
        log_path = f'{self.path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            print(*x)    
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
                       
    def load_args(self, name:str='system_cfg')->SimpleNamespace:
        args = load_json(f'{self.path}/{name}.json')
        return SimpleNamespace(**args)

    @property
    def path(self):
        return f'{BASE_DIR}/{self.exp_name}'
