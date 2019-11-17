#!/usr/bin/env python3
'''model classes for recurrent nets.

Currently, a single recurrent is provided which, for now,
trains only with stochastic gradient descent.

SimpleRNN
  (n_in:int, enc_dim:int, n_hid:int, n_out:int,padding_idx:int)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'Simmons'
__version__ = '0.6'
__status__ = 'Development'
__date__ = '11/17/19'

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

if __name__ == '__main__':
  import doctest
  failures, _ = doctest.testmod()

  if failures == 0:
    # Below prints only the signature of locally defined functions.
    from inspect import signature
    local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
        if callable(ob) and ob.__module__ == __name__]
    for name, ob in local_functions:
      print(name,'\n  ',signature(ob))
