#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from code_funcs import *

def generate_data(data_size, device):
    spiral, color = swiss_roll_data(data_size)
    dataset = torch.tensor(spiral.T).float().to(device)
    print("\nSuccess: loaded spiral data")
    return spiral, dataset, color

def train_spiral(dataset, n_steps, device, batch_size=128, epochs=101, color='red'):
    time_estimation = int(epochs * 0.26764 // 60 + 1)

    print(f"\nTraining the reverse diffusion model on a spiral dataset of size={len(dataset)}\nModel params: n_steps={n_steps}, batch_size={batch_size}, epochs={epochs}\nTraining time estimation: {time_estimation} {'minutes' if time_estimation > 1 else 'minute'}")

    model = train(dataset, n_steps, device, batch_size=batch_size, epochs=epochs, color=color)

    print("\nSuccess: finished training model\n")
    
    return model
    
    
def main(targets):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("\nSuccess: device='cuda'")
    else:
        print("\nWarning: cuda is not available")

    with open('config.json') as fh:
        config = json.load(fh)
    
    data_size = config['spiral']['data_size']
    n_steps = config['spiral']['n_steps']
    batch_size = config['spiral']['batch_size']
    epochs = config['spiral']['epochs']
    
    if 'data' in targets:
        spiral, dataset, color = generate_data(data_size, device)
        
    if 'test' in targets:
        try:
            dataset
        except:
            spiral, dataset, color = generate_data(data_size, device)
            
        model = train_spiral(dataset, n_steps, device, batch_size=batch_size, epochs=101)
        
    if 'all' in targets:
        try:
            dataset
        except:
            spiral, dataset, color = generate_data(data_size, device)
        
        model = train_spiral(dataset, n_steps, device, batch_size=batch_size, epochs=epochs)
        
        if not os.path.exists('models'):
            os.mkdir('models')

        torch.save(model.state_dict(), f"models/model {datetime.datetime.now()} data_size={data_size} n_steps={n_steps} batch_size={batch_size} epochs={epochs}")

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)