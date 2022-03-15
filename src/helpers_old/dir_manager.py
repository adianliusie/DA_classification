import os
import json
import torch
import shutil
import csv

from types import SimpleNamespace
from typing import Callable
from collections import namedtuple

from ..utils import load_json, download_hpc_model
from ..config import config

BASE_DIR = f'{config.base_dir}/trained_models'

class DirManager:
    """ Class which manages logs, models and config files """
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
        """deletes the temp, unsaved experiments directory"""
        if os.path.isdir(f'{BASE_DIR}/temp'): 
            shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        """makes experiments directory"""
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def save_args(self, name:str, args:namedtuple):
        """saves arguments into json format"""
        config_path = f'{self.path}/{name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(args.__dict__, jsonFile, indent=4)

    def make_logger(self, file_name:str)->Callable:
        """creates logging function which saves prints to txt file"""
        log_path = f'{self.path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            print(*x)    
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    def update_curve(self, mode, *args):
        """ logs any passed arguments into a file"""
        with open(f'{self.path}/{mode}.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(args)

    def get_curve(self, mode='train'):
        float_list = lambda x: [float(i) for i in x] 
        with open(f'{self.path}/{mode}.csv') as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            data_read = [float_list(row) for row in reader]
        return tuple(zip(*data_read))
    
    @property
    def path(self):
        """returns base experiment path"""
        return f'{BASE_DIR}/{self.exp_name}'

    @classmethod
    def load_dir(cls, exp_name:str, hpc=False)->'DirManager':
        dir_manager = cls.__new__(cls)
        if hpc: 
            dir_manager.exp_name = 'hpc/'+exp_name
            download_hpc_model(exp_name)
        else:
            dir_manager.exp_name = exp_name

        return dir_manager
    
    def load_args(self, name:str)->SimpleNamespace:
        args = load_json(f'{self.path}/{name}.json')
        return SimpleNamespace(**args)
           
