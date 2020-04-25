import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def which_device ():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

device = which_device()

    
class EncoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, posterior_dim, dropout):
        super(EncoderNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, posterior_dim)
        self.drop = nn.Dropout(p = 0.0)
        
    def forward(self, x, h):
        out, h = self.gru(x,h)
        out1 = F.softmax(self.drop(self.fc(out)), dim=-1)
        return out, out1, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(1, batch_size, self.hidden_dim).zero_().to(device) # 1 is the layer depth
        return hidden

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

    
class Discriminator(nn.Module):
    
    def __init__(self, hidden_dim, number_of_speakers):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, number_of_speakers)


    def forward(self, input_feature, alpha):
        
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.fc1(reversed_input)  
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
    
class DecoderNet(nn.Module):
    def __init__(self, posterior_dim, hidden_dim, output_dim, dropout):
        super(DecoderNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(posterior_dim, hidden_dim)
        self.drop = nn.Dropout(p = dropout)
        self.gru = nn.GRU(hidden_dim, output_dim, batch_first=True)
        
    def forward(self, x, h):
        x = self.drop(self.fc(x))
        out, h = self.gru(x,h)
        # attend here for sure!!!
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(1,batch_size, self.output_dim).zero_().to(device)
        return hidden
    
class pruned_MSE:
    def __call__(self, outputs, labels, lengths):
        x = 0
        for i, output in enumerate(outputs):
            label = labels[i]
            output = output[0:lengths[i]]
            label = label[0:lengths[i]]
            mse = torch.pow((label - output),2).sum()
            x += mse
        divider = float(sum(lengths))*float(output.shape[1])
        return x/divider

class pruned_L2:
    def __call__(self, activations, lengths):
        x = 0
        for i, activation in enumerate(activations):
            activation = activation[0:lengths[i]]
            l2 = torch.pow(activation,2).sum()
            x += l2
        divider = float(sum(lengths))*float(activation.shape[1])
        return x/divider

class masked_MSE:
    def __call__(self, outputs, labels, W, lengths):
        mse = torch.pow((labels - outputs),2)
        mse = torch.mean(mse, dim = -1)
        mse = torch.sum(mse * W)
        mse = mse/float(sum(lengths))
        return mse

class masked_L2:
    def __call__(self, activations, W, lengths):
        l2 = torch.pow(activations,2)
        l2 = torch.mean(l2, dim = -1)
        l2 = torch.sum(l2*W)
        l2 = l2/ float(sum(lengths))
        return l2
    
class FFEncoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, posterior_dim):
        super(FFEncoderNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, posterior_dim)
        self.do = nn.Dropout(0.2)
        
    def forward(self, x):
        out = F.relu(self.do(self.fc1(x)))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out),dim = -1)
        return out
    
    
class FFDecoderNet(nn.Module):
    def __init__(self, posterior_dim, hidden_dim, output_dim):
        super(FFDecoderNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(posterior_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
               
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

