import torch
import argparse
import os
from data import data
from utils import helper_functions as fh
from utils import custom_functions as fc
import model
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from termcolor import colored
from scipy.signal import medfilt
from scipy.ndimage.filters import rank_filter

parser = argparse.ArgumentParser(description='Acoustic Unit Discovery with Correspondence Sparse AutoEncoder')

parser.add_argument('--test_data_dir', type = str, 
                    help = 'location of the data corpus')
parser.add_argument('--num_units', type = int, default=64, metavar ='K',
                     help='number of acoustic units in clustering')
parser.add_argument('--embedding_dim', type=int, default = 128, 
                     help='dimension of the hidden layer(s)')
parser.add_argument('--seed', type=int, default = 42, 
                     help='set seed for reproducibility')
parser.add_argument('--exp_name', type=str, default = 'temp', 
                     help = 'name of the experiment to save the models')

parser.add_argument('--drop_prob', type=float, default=0.0,
                    help='dropout for the decoder')
args = parser.parse_args()
OUTDIR = os.path.join('Results', args.exp_name, 'english/test')
args = parser.parse_args()
device = model.which_device()
fh.random_seed(args.seed)

DATA_DIM = 16
PLP_POWER=0.90

encoder = model.EncoderNet(DATA_DIM, args.embedding_dim, args.num_units, dropout=args.drop_prob).to(device)
model_file = args.exp_name + '.pth'
encoder.load_state_dict(torch.load(model_file))
encoder.to(device)

print(' extraction started for test')


if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

        
files = os.listdir(args.test_data_dir)
ctr = 1
for file in files:
    print(str(ctr) + '/' + str(len(files)), end='\r')
    ctr+= 1
    filename = os.path.join(args.test_data_dir, file)
    test_utt = np.loadtxt(filename)
    if len(test_utt.shape) == 1:
       test_utt = test_utt.reshape(1, -1)
    current_max = -1
    matrix = torch.from_numpy(test_utt).to(device).float()
    matrix = matrix.view(1,test_utt.shape[0],test_utt.shape[-1])

    h = encoder.init_hidden(matrix.shape[0])
    h = h.data
    _, encoded, h = encoder(matrix,h)
    encoded = fc.wta_net(encoded)     

    encoded = encoded.view(test_utt.shape[0], args.num_units)
    encoded = encoded.data.to('cpu').numpy()
    post = np.transpose(encoded)
    new_post = np.zeros(post.shape)
    
    for state in range(post.shape[0]):
        new_frame = medfilt(post[state,:],3)
        new_post[state,:] = new_frame
    
    encoded = np.transpose(new_post)    
    filename = os.path.join(OUTDIR, file[:-4] + '.txt')
    f2 = open(filename, 'a')
    for line in range(len(encoded)):
        vect = encoded[line]
        onehot = np.zeros(args.num_units)
        onehot[np.argmax(vect)] = 1
        if current_max == np.argmax(vect): 
            continue
        current_max = np.argmax(vect) 
        for feature in range(args.num_units-1):
            tmp2 = onehot[feature]
            f2.write("%d " % tmp2)
        tmp2 = onehot[args.num_units-1]
        f2.write("%d" % tmp2)
        f2.write('\n')        
    f2.close()


