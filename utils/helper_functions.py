import numpy as np
import os
from termcolor import colored
import torch
import random
import pandas as pd
from random import choice

def random_seed(seed_value):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars 
    random.seed(seed_value) # Python 
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value) # gpu vars 
    torch.backends.cudnn.deterministic = True #needed 
    torch.backends.cudnn.benchmark = False
    
def create_dataset(dataset, look_back=150, step = 75):
    
    dataX = np.zeros(((len(dataset)-look_back)//step,look_back,dataset.shape[1]))
    for i in range(0,dataX.shape[0]):
        dataX[i] = dataset[i*step : i*step + look_back, : ]
    dataX = dataX.reshape(-1,look_back,dataset.shape[1])
    return dataX


def create_dataset_for_labels(dataset, look_back=100, step = 75):
    
    dataX = np.zeros(((len(dataset)-look_back)//step,look_back))
    for i in range(dataX.shape[0]):
        dataX[i] = dataset[i*step : i*step + look_back]
    dataX = dataX.reshape(-1,look_back,1)
    return dataX
