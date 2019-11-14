#!/usr/bin/env python3
'''Model classes for convolutional and recurrent nets.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from du.wordlib import unpad_sequence

__author__ = 'Simmons'
__version__ = '0.3'
__status__ = 'Development'
__date__ = '11/14/19'

class OneMetaCNN(nn.Module):
  ''' A one meta-layer convolutional model.
  '''

  def __init__(self, means = None, stdevs = None):
    super(OneMetaCNN, self).__init__()
    self.meta_layer1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )
    self.fc_layer1 = nn.Linear(1600,10)

    # recommended hyper-parameters (with seed 123, epochs 30,
    # centered and normalized)
    self.register_buffer('lr', torch.tensor(.023))
    self.register_buffer('mo', torch.tensor(.9))
    self.register_buffer('bs', torch.tensor(30))

    # Add mean and stdev fields to state_dict.  These hold
    # the means and stdevs of the training data.
    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = torch.reshape(xss, (-1, 1600))
    xss = self.fc_layer1(xss)
    return torch.log_softmax(xss, dim=1)

class TwoMetaCNN(nn.Module):
  ''' A two meta-layer convolutional model.
  '''

  def __init__(self, width_m1=16, width_m2=32, means = None, stdevs = None):
    super(TwoMetaCNN, self).__init__()
    self.width_m2 = width_m2
    self.meta_layer1 = nn.Sequential( # A mini-batch_size of N for input to this
        nn.Conv2d(                    # would have size Nx1x20x20.
            in_channels=1,
            out_channels=width_m1,  # And the output of Conv2d is size:
            kernel_size=5,            #     N x width_m1 x 20 x 20.
            stride=1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )                                 # Downsampling with MaxPool we have that
    self.meta_layer2 = nn.Sequential( # the input here is:
        nn.Conv2d(                    #     N x width_m1 x 10 x 10.
            in_channels=width_m1,
            out_channels=width_m2,
            kernel_size=3,            # And the ouput of this Conv2d is:
            stride=1,                 #     N x width_m2 x 10 x 10.
            padding = 1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )                                 # Downsampling, we have
                                      #   N x width_m2 x 5 x 5.
    self.fc_layer1 = nn.Linear(width_m2*25,200)
    self.fc_layer2 = nn.Linear(200,10)

    # recommended hyper-parameters
    self.register_buffer('lr', torch.tensor(1e-4))
    self.register_buffer('mo', torch.tensor(.95))

    # Add mean and stdev fields to state_dict.  These hold
    # the means and stdevs of the training data.
    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = self.meta_layer2(xss)
    xss = torch.reshape(xss, (-1, self.width_m2*25))
    xss = self.fc_layer1(xss)
    xss = torch.relu(xss)
    xss = self.fc_layer2(xss)
    return torch.log_softmax(xss, dim=1)

class SimpleRNN(nn.Module):
  def __init__(self, n_in, enc_dim, n_hid, n_out, padding_idx, device = 'cpu'):
    super(SimpleRNN, self).__init__()
    self.n_in = n_in
    self.n_hid = n_hid
    self.device = device
    self.padding_idx = padding_idx
    self.hidden = torch.zeros(1, n_hid).to(device)
    self.comb2hid = nn.Linear(n_in + n_hid, n_hid)
    self.comb2out = nn.Linear(n_in + n_hid, n_out)

  def forward(self, xss, lengths = None):
    xs = xss.squeeze(0)[:lengths.item()]
    for x_ in xs:
      x_one_hot = F.one_hot(x_, self.n_in).float().unsqueeze(0)
      combined = torch.cat((x_one_hot, self.hidden),dim = 1)
      self.hidden = self.comb2hid(combined)
    logit = self.comb2out(combined)
    self.hidden = torch.zeros(1, self.n_hid).to(self.device)
    return torch.log_softmax(logit,dim=1)
