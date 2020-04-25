"""

Train the Correspondence Recurrent Sparse Autoencoder (CoRSA) model.

In addition to CoRSA training, by default, the following are employed:

1. Temporal winner-take-all (twta)
2. Vector quantization on the intermediate layer
3. Speaker adversarial training

author : Batuhan Gundogdu

"""
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


parser = argparse.ArgumentParser(description='Acoustic Unit Discovery with Correspondence Sparse AutoEncoder')

parser.add_argument('--data_dir', type = str, 
                    help = 'location of the data corpus')
parser.add_argument('--utd_pairs_dir', type = str, 
                    help = 'location of the UTD pairs, file.dedups and file.nodes')
parser.add_argument('--reg_weight', type=float, default=1.0,
                    help='weight of the L2-norm cost of the intermediate layer')
parser.add_argument('--batch_size', type = int, default = 1024, metavar='N',
                     help='batch size for AE and SAT training')
parser.add_argument('--batch_size_corsa', type = int, default = 256, metavar='N2',
                     help='batch size for the correspondence training')
parser.add_argument('--num_units', type = int, default=64, metavar ='K',
                     help='number of acoustic units in clustering')
parser.add_argument('--embedding_dim', type=int, default = 128, 
                     help='dimension of the hidden layer(s)')
parser.add_argument('--epochs', type = int, default=10, 
                     help='number of epochs')
parser.add_argument('--seed', type=int, default = 42, 
                     help='set seed for reproducibility')
parser.add_argument('--timesteps', type=int, default=250, metavar ='T', 
                     help='sequence length inputted to RNN')
parser.add_argument('--timesteps_corsa', type=int, default=80, metavar ='T2', 
                     help='sequence length inputted to RNN for correspondence training')
parser.add_argument('--exp_name', type=str, default = 'temp', 
                     help = 'name of the experiment to save the models')
parser.add_argument('--learn_rate', type=float, default=0.001,
                    help='learning rate of ae')
parser.add_argument('--learn_rate_corsa', type=float, default=0.0005,
                    help='learning rate of sat and corsa')
parser.add_argument('--drop_prob', type=float, default=0.0,
                    help='dropout for the decoder')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='weight of adversarial backprob')

PLP_POWER = 0.90
DATA_DIM = 16
args = parser.parse_args()
fh.random_seed(args.seed)
device = model.which_device()

opt = vars(args)
print('loading the data from folder')
dataset, data_dict, speaker_ids = data.load(opt['data_dir'])

num_speakers = max(speaker_ids) # Although it works for now, this might not be the case, check later

dataset = dataset[:, 0:DATA_DIM]
for i in range(DATA_DIM):
    dataset[:, i]=dataset[:, i]*pow(PLP_POWER, i)


if not (os.path.exists(os.path.join(opt['utd_pairs_dir'], 'pair1.npy')) 
        and os.path.exists(os.path.join(opt['utd_pairs_dir'], 'pair2.npy')) ):
    
    print('pairs dont exist, getting them via matching the corresponding pairs obtained from UTD')
    corr_a, corr_b = data.create_corr_data(opt, data_dict)
    
else : 
    print('pairs exist in the corr directory. using them.')
    corr_a = np.load(os.path.join(opt['utd_pairs_dir'], 'pair1.npy'))
    corr_b = np.load(os.path.join(opt['utd_pairs_dir'], 'pair2.npy'))
    

dataset = fh.create_dataset(dataset, args.timesteps, args.timesteps)
corr_a = fh.create_dataset(corr_a, args.timesteps_corsa, args.timesteps_corsa) 
corr_b = fh.create_dataset(corr_b, args.timesteps_corsa, args.timesteps_corsa) 
speaker_ids = fh.create_dataset_for_labels(speaker_ids, args.timesteps, args.timesteps)

val_portion = 0.1
val_data_size = int(val_portion*len(dataset))
train_x = dataset[:-val_data_size]
train_sp = speaker_ids[:-val_data_size]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_x), torch.from_numpy(train_sp))
train_loader = DataLoader(train_data, shuffle = True, batch_size = args.batch_size, drop_last = True) 

corr_data = TensorDataset(torch.from_numpy(corr_a).to(device), torch.from_numpy(corr_b).to(device))
corr_loader = DataLoader(corr_data, shuffle = True, batch_size = args.batch_size_corsa, drop_last = True)



print(len(corr_loader), len(train_loader))


encoder = model.EncoderNet(DATA_DIM, args.embedding_dim, args.num_units, dropout=args.drop_prob).to(device).train()
decoder = model.DecoderNet(args.num_units, args.embedding_dim, DATA_DIM, dropout=args.drop_prob).to(device).train()
discriminator = model.Discriminator(args.embedding_dim, num_speakers).to(device).train()

params = list(encoder.parameters()) + list(decoder.parameters()) 

optimizer = torch.optim.Adam(params, lr = args.learn_rate)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr = args.learn_rate_corsa)
optimizer_enc = torch.optim.Adam(encoder.parameters(), lr = args.learn_rate_corsa)
optimizer_corsa = torch.optim.Adam(params, lr = args.learn_rate_corsa)

decoder_criterion = torch.nn.MSELoss()
discriminator_criterion = torch.nn.CrossEntropyLoss()


for epoch in range(args.epochs) :
    if (epoch < 4):
        print(colored("AE training",'green'))
        for x, y, _  in train_loader:
            fc.train_ae(x, y, 
                     encoder, decoder, 
                     decoder_criterion,
                     args.reg_weight, optimizer, args.batch_size) 
    else :
        if (epoch%3 == 0):
            print(colored("SA training",'magenta'))
            #for x, _, z  in train_loader:

                
                #fc.train_disc(x, z, encoder, discriminator, 
                #         discriminator_criterion,optimizer_disc, optimizer_enc, args.batch_size, args.alpha)        
        elif (epoch%3 == 1):       
        
            print(colored("a -> b",'red'))
            for x, y  in corr_loader:
                fc.train_ae(x, y, encoder, decoder, decoder_criterion, args.reg_weight, optimizer_corsa, args.batch_size_corsa)        
        elif (epoch%3 == 2): 
            print(colored("b -> a",'blue'))
            for x, y  in corr_loader:
                fc.train_ae(y, x, encoder, decoder, decoder_criterion, args.reg_weight, optimizer_corsa, args.batch_size_corsa) 

encoder_PATH = args.exp_name + '.pth'                
torch.save(encoder.state_dict(), encoder_PATH)
print('done training')
print('model saved in ' + encoder_PATH)


