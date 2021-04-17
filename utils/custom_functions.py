import torch
import torch.nn.functional as F
from torch.autograd import Function
import os
import numpy as np

class ArgmaxWithGradientPassBackward(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        dim = -1  # Set in stone for now because one_hot function only operates on the last dimension
        result = input_tensor.argmax(dim=dim)
        return torch.nn.functional.one_hot(result, input_tensor.size(dim)).float()

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs
    
def wta_net (x):
    
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    num_units = x.shape[2]
    wta1 = -np.ones((num_units,num_units))

    for i in range(num_units):
        wta1[i][i] = num_units-1

    wta1 = torch.from_numpy(wta1).to(device).float()    

    wta2 = np.eye(num_units)

    wta2 = torch.from_numpy(wta2).to(device).float()    

    q1 = x.matmul(wta1)
    q2 = x.matmul(wta2)
    result1 = torch.from_numpy(np.zeros((x.shape[0],x.shape[1],x.shape[2]+1))).to(device).float()
    result2 = torch.from_numpy(np.zeros((x.shape[0],x.shape[1],x.shape[2]+1))).to(device).float()
    result1[:,:,:x.shape[2]] = q2
    result2[:,:,1:] = q1
    result = result1 + result2
    result3 = result[:,:,:x.shape[2]]
    return F.softmax(F.relu(result3), dim = -1) 

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def predict(seg_test_data, encoder, keys, pred_dir, num_units, language = 'english'):

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    for utt in seg_test_data:
        current_max = -1
        matrix = torch.from_numpy(seg_test_data[utt]).to(device).float()
        matrix = matrix.view(1,seg_test_data[utt].shape[0],seg_test_data[utt].shape[1])
        
        h = encoder.init_hidden(matrix.shape[0])
        h = h.data
        _, encoded, h = encoder(matrix,h)
        encoded = wta_net(encoded)     

        encoded = encoded.view(seg_test_data[utt].shape[0], num_units)
        encoded = encoded.data.to('cpu').numpy()
        filename = keys[utt]
        filename = pred_dir + filename[:-4] + '.txt' if language == 'english' else pred_dir + filename + '.txt'
        f2 = open(filename, 'a')
        for line in range(len(encoded)):
            vect = encoded[line]
            onehot = np.zeros(num_units)
            onehot[np.argmax(vect)] = 1
            if current_max == np.argmax(vect): # current maximum feature is the same as the previous
                continue # do not execute the following lines (writing to the norep files)
            current_max = np.argmax(vect) # new maximum is at another feature number
            for feature in range(num_units-1):
                tmp2 = onehot[feature]
                f2.write("%d " % tmp2)
            tmp2 = onehot[num_units-1]
            f2.write("%d" % tmp2)
            f2.write('\n')        
        f2.close()
        
def predict_unc(seg_test_data, encoder, keys, pred_dir, num_units, language = 'english'):
    
        
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    for utt in seg_test_data:
        current_max = -1
        matrix = torch.from_numpy(seg_test_data[utt]).to(device).float()
        matrix = matrix.view(1,seg_test_data[utt].shape[0],seg_test_data[utt].shape[1])
        
        h = encoder.init_hidden(matrix.shape[0])
        h = h.data
        encoded, h = encoder(matrix,h)
        encoded = temporal_wta_net(encoded,wta1,wta2)     

        encoded = encoded.view(seg_test_data[utt].shape[0],num_units)
        encoded = encoded.data.to('cpu').numpy()
        filename = keys[utt]
        filename = pred_dir + filename[:-4] + '.txt' if language == 'english' else pred_dir + filename + '.txt'
        f2 = open(filename, 'a')
        np.savetxt(f2, encoded, fmt='%.6f')    
        f2.close()
        


def train_ae(x, y, encoder, decoder, mse, reg_weight, optimizer, batch_size):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_for_encoder = encoder.init_hidden(batch_size)
    hidden_for_decoder = decoder.init_hidden(batch_size)
    hidden_for_encoder = hidden_for_encoder.data
    hidden_for_decoder = hidden_for_decoder.data
    encoder.zero_grad()
    decoder.zero_grad()
    _, posterior_activations, hidden_for_encoder = encoder(x.to(device).float(), hidden_for_encoder)
    ### WINNER TAKE ALL
    quantized = ArgmaxWithGradientPassBackward.apply(wta_net(posterior_activations))     
    reconstructed, hidden_for_decoder = decoder(quantized.float(), hidden_for_decoder)

    loss1 = torch.mean(torch.pow(quantized, 2))
    loss2 = mse(reconstructed, y.to(device).float())
    total_loss = loss2 - reg_weight*loss1
    total_loss.backward()
    optimizer.step()

    
    
    
def train_disc(x, z, 
             encoder, discriminator, discriminator_criterion, 
             optimizer_disc, optimizer_enc, batch_size, alpha):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_for_encoder = encoder.init_hidden(batch_size)
    hidden_for_encoder = hidden_for_encoder.data
    optimizer_enc.zero_grad()
    optimizer_disc.zero_grad()
    embedding, posterior_activations, hidden_for_encoder = encoder(x.to(device).float(), hidden_for_encoder)
    
    speaker_prediction = discriminator(embedding, alpha)
    speaker_prediction = speaker_prediction.view(-1, speaker_prediction.shape[-1])
    speaker_label = z.view(-1).long()
    discriminator_loss = discriminator_criterion(speaker_prediction, speaker_label.to(device))
    
    discriminator_loss.backward()
    optimizer_enc.step()
    optimizer_disc.step()

def train_sa(x, y, z, 
             encoder, decoder, discriminator, 
             decoder_criterion, discriminator_criterion, 
             reg_weight, optimizer, optimizer_disc, batch_size, alpha):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_for_encoder = encoder.init_hidden(batch_size)
    hidden_for_decoder = decoder.init_hidden(batch_size)
    hidden_for_encoder = hidden_for_encoder.data
    hidden_for_decoder = hidden_for_decoder.data
    optimizer.zero_grad()
    optimizer_disc.zero_grad()
    embedding, posterior_activations, hidden_for_encoder = encoder(x.to(device).float(), hidden_for_encoder)
    quantized = ArgmaxWithGradientPassBackward.apply(wta_net(posterior_activations))     
    reconstructed, hidden_for_decoder = decoder(quantized.float(), hidden_for_decoder)
    
    speaker_prediction = discriminator(embedding, alpha)
    speaker_prediction = speaker_prediction.view(-1, speaker_prediction.shape[-1])
    speaker_label = z.view(-1)
    discriminator_loss = discriminator_criterion(speaker_prediction, speaker_label.to(device).long())
    decoder_loss = decoder_criterion(reconstructed, y.to(device).float())
    L2_loss = torch.mean(torch.pow(quantized, 2))
    decoder_loss = decoder_criterion(reconstructed, y.to(device).float())
    total_loss = decoder_loss - reg_weight*L2_loss + discriminator_loss
    total_loss.backward()
    optimizer.step()
    optimizer_disc.step()
    
    return decoder_loss.item(), discriminator_loss.item()
