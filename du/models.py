#!/usr/bin/env python3
"""classes and functions for ~feed-forward neural nets~.

\n`QUICK SIGNATURES`

  |Net_|
    |FFNet_|($means$=None, $stdevs$=None)
      |SimpleLinReg|($degree$=1, $**kwargs$)
      |DenseFFNet|($n_inputs$, $n_outputs$, $widths$, $**kwargs$)

  |denseFFhidden|($n_inputs$, $n_outputs$, $widths$, $**kwargs$)
  |polyize|($xss$, $degree$)

The base class for all neural nets is `Net_` which subclasses
`nn.Module` and implements a single method `numel_` that returns
the number of ~parameters~ (i.e. ~weights~ of the corresponding
model).

The class `FFNet_` is a base class for feed forward neural nets.
It is a subclass of `nn.Module` that adds the single feature that
its constructor allows passing in attributes called `means` and
`stdevs` which we use in practice to store the ~means~ and ~stan-~
~dard deviations~ of our ~training data~.

The reason that we might want to store the training data means
and standard deviations as attributes in `FFNet_` is that they
will then be saved (along with the weights) upon ~serialization~
of a trained instance of the model class.

Then, when we later want to use the pre-trained model to make
a ~prediction~, we can read in the serialized model and easily
~center~ and/or ~normalize~ (if in fact we pre-applied either or
both of those processes to the training data) the ~features~ of
the prediction with respect to the means and standard devia-
tions of the training data.

If we serialized, in this way, the means and/or stdevs of the
training data, then we typically don't even need the training
data when making predictions.

The other classes and helper functions that are defined in this
module can be briefly described as follows:

`SimpleLinReg` can be used in conjuction with its helper function
`polyize` to execute ~simple polynomial regression~, where 'simple'
means that the both the features and the targets of each example
are real numbers.

Said differently, polynomial regression means fitting a polyno-
mial to a point cloud. By simple polynomial regression we mean
fitting a polynomial to a point cloud that lives in `R^2`. Type
`pd du.examples` at your command-line to see demonstation of us-
ing `SimpleLinReg` to perform polyomial regression.

`DenseFFNet` is a factory class for building dense (i.e., fully
connected) feed-forward neural nets. The function `denseFFhidden`
is a helper function that can be used to build the hidden part
of a dense, feedforward net or, more generally, the trailing
dense part of, say, a convoutional net.
"""
#Todo:
#  DenseFF doesn't specialize to no hidden layers. <--DONE??

from textwrap import dedent
import torch
import torch.nn as nn
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

def denseFFhidden(n_inputs, n_outputs, widths, **kwargs):
  """Return `ModuleDict` for a dense chunk of a feed-forward net.

  Returns an instance of `nnModuleDict` determining the fully-
  connected composition of linear layers with the specified
  widths and nonlinearities.

  Args:
    $n_inputs$ (`int`): Number of inputs into the first layer.
    $n_outputs$ (`int`): Number of outputs from the last layer.
    $widths$ (`Tuple[int]`): The widths (i.e., number of nodes)
        in the successive layers.

  Kwargs:
    $nonlins$ (`Tuple[nn.Module]`): The nonlinearities for each
        layer. If this has length 1, then use that nonlinearity
        for each hidden layer. Default: `(nn.ReLU(),)`.
    $dropout$ (`float`): If greater than zero, add a dropout layer
        with this probablity before each nonlinearity. Def: `0`.

  Returns:
    `nn.ModuleDict`.

  >>> `print(denseFFhidden(1, 1, (16, 32)))`
  Sequential(
    (0): Linear(in_features=1, out_features=16, bias=True)
    (act0): ReLU()
    (lin1): Linear(in_features=16, out_features=32, bias=True)
    (act1): ReLU()
    (lin2): Linear(in_features=32, out_features=1, bias=True)
  )
  >>> `print(denseFFhidden(10, 1, (8,), nonlins=(nn.Tanh(),)))`
  Sequential(
    (0): Linear(in_features=10, out_features=8, bias=True)
    (act0): Tanh()
    (lin1): Linear(in_features=8, out_features=1, bias=True)
  )
  """
  du.utils._check_kwargs(kwargs, ['nonlins','dropout'])
  nonlins = kwargs.get('nonlins', tuple([nn.ReLU()]))
  dropout = kwargs.get('dropout', 0)
  assert isinstance(nonlins, (tuple, list)), dedent("""\
      'nonlins should be a tuple or a list. not a {}
  """.format(type(nonlins)))
  for nonlin in nonlins:
    assert isinstance(nonlin,nn.Module), dedent("""\
        'the items in nonlins should be instances of nn.Module, not {}
    """.format(type(nonlin)))
  if len(nonlins) == 1:
    nonlins = tuple(list(nonlins) * len(widths))

  widths = list(widths) + [n_outputs]
  block = nn.Sequential(nn.Linear(n_inputs, widths[0]))
  for layer in range(len(widths)-1):
    if dropout > 0:
      block.add_module('dropout', nn.Dropout2d(p=dropout))
    block.add_module('act'+str(layer),nonlins[layer])
    block.add_module(
        'lin'+str(layer+1),
        nn.Linear(widths[layer], widths[layer+1]))
  return block

class Net_(nn.Module):
    """A base class for all of DUlib's neural nets.

    """
    def __init__(self):
        super().__init__()

    def numel_(self, trainable=True):
        """Return the number of parameters."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

class FFNet_(Net_):
  """A base class for feed-forward neural nets.

  This simply adds to `nn.Module` attributes called `means` and `st`
  `devs`. Those can be useful when serializing a model since one
  can store, along with the trained weights, the means and stan-
  dard deviations with which one must standardize when making
  predictions.

  Note: when unserializing, one re-instantiates the same model
  that was used for training. When doing so, pass any tensors
  of the correct size to `means` and `stdevs`.
  """
  def __init__(self, means = None, stdevs = None):
    """Constructor.

    Args:
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data. Default: `None`.
      $stdevs$ (`torch.Tensor`): A tensor typically holding the
          standard deviations of the training data. Default:
          `None`.
    """
    super().__init__()
    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

def polyize(xss, degree):
  """Return a tensor suitable for poly regression.

  This is essentially a helper function when computing polynom-
  ial (with `degree` > 1) regression. The return tensor is just
  the ~Vandermonde matrix~ without the leading column of ones.

  Args:
    $xss$ (`tensor`): A tensor of shape `(n,1)`.
    $degree$ (`int`): The degree of the desired regressing poly.

  Returns:
    `tensor`. A tensor of shape `(n,degree)` in which the first co-
        lumn is `xss`, the second holds the squares of the elements
        of `xss`, etc.

  >>> `xss = torch.arange(1.,4.).unsqueeze(1)`
  >>> `polyize(xss, 2)`
  tensor([[1., 1.],
          [2., 4.],
          [3., 9.]])
  >>> `xss = torch.rand(5).unsqueeze(1)`
  >>> `torch.all(xss == polyize(xss, 1)).item()`
  1
  """
  assert xss.dim() == 2 and xss.size()[1] == 1
  #copy xss to the cols of a (len(xss),deg) tensor
  new = xss * torch.ones(len(xss), degree)
  #square entries in 2rd col, cube those in the 3rd,...
  return new.pow(torch.arange(1., degree+1))

class SimpleLinReg(FFNet_):
  """A class for simple polynomial regression.

  This is for fitting a polynomial of specified degree to a
  point cloud in R^2. Here, the mathematical model is nonlin-
  ear in the inputs `x` (if `degree > 1) but linear in the
  weights.
  """
  def __init__(self, degree=1, **kwargs):
    """Constructor

    Args:
      $degree$ (`int`): Fit with a poly of this degree.  Default:
      `1`.

    Kwargs:
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data. Default: `None`.
      $stdevs$ (`torch.Tensor`): A tensor typically holding
          the standard deviations of the training data. Def-
          ault: `None`.
    """
    assert degree > 0, 'just take the means of the ys'
    du.utils._check_kwargs(kwargs,['means','stdevs'])
    means=kwargs.get('means',None); stdevs=kwargs.get('stdevs',None)
    super().__init__(means = means, stdevs = stdevs)
    self.layer = nn.Linear(degree, 1)
  def forward(self, xss):
    """Forward pass tensor through the model.

    Args:
      $xss$ (`torch.Tensor`): A tensor holding the feat-
          ures of shape `(n,degree)` where `n` is the
          number of examples.

    Returns:
      `torch.Tensor`. A tensor with shape `(n,1)`.
    """
    return self.layer(xss)

class DenseFFNet(FFNet_):
  """Fully-connected, feed-forward net."""

  def __init__(self, n_inputs, n_outputs, widths=(), **kwargs):
    """Constructor.

    Args:
      $n_inputs$ (`int`): No. of features in each example.
      $n_outputs$ (`int`): No. of targets in each example.
      $widths$ (`Tuple[int]`): The width (i.e., the no. of nodes
          in each successive hidden layer. If the tuple is
          empty, then an instance of this will have no hid-
          den layers. Default: `()`.

    Kwargs:
      $nonlins$ (`Tuple[nn.Module]`): The nonlinearities for
          each hidden layer. If this has length 1, then use
          that nonlinearity for each hidden layer.  Default:
          `(nn.ReLU(),)`.
      $dropout$ (`float`): If greater than zero, add a dropout
          layer with this probablity before each nonlinear-
          ity. Default: `0`.
      $outfn$ (`nn.Module`): a function to pipe out through
          lastly in the `forward` method; e.g.,
              `lambda xss: log_softmax(xss, dim=1)`.
          (Here we used a lambda function solely so that we
          can provide the required `dim=1` to Torch's `log_`
          `softmax`). Default: `None`.
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data. Default: `None`.
      $stdevs$ (`torch.Tensor`): A tensor typically holding
          the standard deviations of the training data. Def-
          ault: `None`.
    """
    du.utils._check_kwargs(kwargs,['nonlins','outfn','means','stdevs','dropout'])
    means=kwargs.get('means',None); stdevs=kwargs.get('stdevs',None)
    super().__init__(means = means, stdevs = stdevs)
    nonlins = kwargs.get('nonlins', tuple([nn.ReLU()]))
    dropout = kwargs.get('dropout', 0)
    self.outfn = kwargs.get('outfn', None)
    assert isinstance(widths, (tuple, list)), dedent("""\
        widths should be a tuple or a list, not a {}
    """.format(type(widths)))
    assert isinstance(nonlins, (tuple, list)), dedent("""\
        'nonlins should be a tuple or a list. not a {}
    """.format(type(nonlins)))
    for nonlin in nonlins:
      assert isinstance(nonlin,nn.Module), dedent("""\
          'the items in nonlins should be instances of nn.Module, not {}
      """.format(type(nonlin)))
    if len(nonlins) == 1:
      nonlins = tuple(list(nonlins) * len(widths))
    assert len(widths) == len(nonlins),dedent("""\
        len(nonlins) (which is {}) must be 1 or must have length
        equal to len(widths) (which is {})
    """.format(len(widths),len(nonlins)))
    self.model = denseFFhidden(
        n_inputs, n_outputs, widths, nonlins=nonlins, dropout=dropout)

  def forward(self, xss):
    """Forward pass tensor through the model.

    Args:
      $xss$ (`tensor`): The features of shape `(n,m)` where `n` is the
          number of examples and `m` is the number of numbers in
          each example's features.

    Returns:
      `torch.Tensor`. The returned tensor has shape `(n,k)` where
          `n` is the number of examples and `k` is the number of
          numbers in each example's targets.
    """
    xss = self.model(xss)
    if self.outfn: xss = self.outfn(xss)
    return xss

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
