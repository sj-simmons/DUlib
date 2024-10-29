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
import du.utils

__author__ = 'Scott Simmons'
__version__ = '0.9.96'
__status__ = 'Development'
__date__ = '10/28/24'
__copyright__ = """
  Copyright 2019-2025 Scott Simmons

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
__license__= 'Apache 2.0'

class SimpleRNN(nn.Module):
  def __init__(self, n_in, enc_dim, n_hid, n_out, padding_idx, device = 'cpu'):
    super().__init__()
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
  failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)

  if failures == 0:
    # Below prints only the signature of locally defined functions.
    from inspect import signature
    local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
        if callable(ob) and ob.__module__ == __name__]
    for name, ob in local_functions:
      print(name,'\n  ',signature(ob))

if __name__ == '__main__':
  import inspect
  import doctest

  # find the user defined functions
  _local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
       if callable(ob) and ob.__module__ == __name__]

  #remove markdown
  #  from the docstring for this module
  globals()['__doc__'] = du.utils._markup(globals()['__doc__'],strip = True)
  #  from the functions (methods are fns in Python3) defined in this module
  for _, _ob in _local_functions:
    if inspect.isfunction(_ob):
      _ob.__doc__ = du.utils._markup(_ob.__doc__,strip = True)
    # below we find all the methods that are not inherited
    if inspect.isclass(_ob):
      _parents = inspect.getmro(_ob)[1:]
      _parents_methods = set()
      for _parent in _parents:
        _members = inspect.getmembers(_parent, inspect.isfunction)
        _parents_methods.update(_members)
      _child_methods = set(inspect.getmembers(_ob, inspect.isfunction))
      _child_only_methods = _child_methods - _parents_methods
      for name,_meth in _child_only_methods:
        _ob.__dict__[name].__doc__ = du.utils._markup(_meth.__doc__,strip =True)

  # run doctests
  failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)

  # print signatures
  if failures == 0:
    from inspect import signature
    for name, ob in _local_functions:
      print(name,'\n  ', inspect.signature(ob))
