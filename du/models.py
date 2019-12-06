#!/usr/bin/env python3
'''model classes for ~feed-forward neural nets~.

The class `FFNN_` defined below is a base class for feed for-
ward neural nets. It is a subclass of `nn.Module` that adds
the single feature that its constructor allows passing in
attributes called `means` and `stdevs` which we use in practice
to store the ~means~ and ~standard deviations~ of our ~training~
~data~.

The reason that we sometimes want to store the training data
means and standard deviations as attributes in `FFNN_` is that
they will then be saved (along with the weights) upon ~serial-~
~ization~ of a trained instance of the class.

Then, when we later want to use the pre-trained model to make
a ~prediction~, we can read in the serialized model and easily
~center~ and/or ~normalize~ (if we in fact we pre-applied either
or both of those processes to the training data) the ~features~
of the prediction with respect to the means and centers of the
training data.  Since we serialized the means and/or stdevs of
the training data, we don't even need the training data when
making predictions.
                     _____________________

Three classes and helper function are defined here.

The class `PolyReg` is suitable for ~simple~ ~polynomial~ ~regres-~
~sion~; where the ~simple~ means that the both the features and
the targets of each example live in the real number line; i.e.,
the inputs and outputs are both 1-dimensional.

Said differently, polynomial regression means fitting a poly-
nomial to point cloud. By simple polynomial regression we mean
fitting a polynomial to a point cloud that lives in `R^2`.

Two generic fully connected model classes, one with a single
extensible width hidden layer and another with two such hidden
layers are also defined.
'''
#Todo:
#  DenseFF doesn't specialize to no hidden layers.

from textwrap import dedent
import torch
import torch.nn as nn
import torch.nn.functional as F
import du.util

__author__ = 'Scott Simmons'
__version__ = '0.8.5'
__status__ = 'Development'
__date__ = '12/06/19'
__doc__ = du.util._markup(__doc__)

def polyize(xss, degree):
  '''Return a tensor suitable for poly regression.

  This is essentially a helper function when computing
  polynomial (with `degree` > 1) regression.

  Args:
    xss (tensor): A tensor of shape `(n,1)`.

  Returns:
    tensor. A tensor of shape `(n,degree)` in which the first
       column is `xss`, the second has the elements of `xss`
       squared, etc.

  >>> xss = torch.arange(1.,4.).unsqueeze(1)
  >>> polyize(xss, 2)
  tensor([[1., 1.],
          [2., 4.],
          [3., 9.]])
  >>> xss = torch.rand(5).unsqueeze(1)
  >>> torch.all(xss == polyize(xss, 1)).item()
  True
  '''
  assert xss.dim() == 2 and xss.size()[1] == 1
  #copy xss to the cols of a (len(xss),deg) tensor
  new = xss * torch.ones(len(xss), degree)
  #square entries in 2rd col, cube those in the 3rd,...
  return new.pow(torch.arange(1., degree+1))
polyize.__doc__= du.util._markup(polyize.__doc__)

class SimpleLinReg(nn.Module):
  """A class for simple linear regression.

  This is for fitting a polynomial of specified degree to a
  point cloud in R^2.
  """
  def __init__(self, degree):
    """Constructor

    Args:
      `degree` (int): Fit a poly of this degree.
    """
    super().__init__()
    self.layer = nn.Linear(degree, 1)
  def forward(self, xss):
    """Forward pass tensor through the model.

    Args:
      xss (tensor): The features of shape `(n,degree)` where
          `n` is the number of examples.

    Returns:
      tensor. The returned tensor has shape `(n,1)`.
    """
    return self.layer(xss)

class FFNNet_(nn.Module):
  """A base class for feed-forward neural nets.

  This simply adds to `nn.Module` attributes called
  `means` and `stdevs`
  """
  def __init__(self, means = None, stdevs = None):
    """Constructor.

    Args:
      `means` (torch.Tensor): A tensor typically holding
          the means of the training data.
      `stdevs` (torch.Tensor): A tensor typically holding
          the standard deviations of the training data.
    """
    super().__init__()
    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

class DenseFF(FFNNet_):
  """Fully-connected feed-forward net."""

  def __init__(self, n_inputs, n_outputs, widths, **kwargs):
    """Constructor.

    Args:
      n_inputs (int): Number of features of each example.
      n_outputs (int): Number of targets of each example.
      widths (Tuple[int]): The width of (i.e., number of
          nodes in the hidden layer. If empty, then an
          instance of this will have no hidden layers.
          Default: ().

    Kwargs:
      nonlins (Tuple[nn.Module]): The nonlinearities for
          each hidden layer. If this has length 1, then
          use that nonlinearity for each hidden layer.
          Default: (`nn.ReLU()`).
      outfn (nn.Module): a function to pipe out though
          lastly in the `forward` method; For example,
          `lambda outss: log_softmax(outss, dim=1)`. Here
          we used a lambda function only so that we could
          provide the required `dim=1` to `log_softmax`.
          Default: `None`.
      means (torch.Tensor): A tensor typically holding
          the means of the training data.
      stdevs (torch.Tensor): A tensor typically holding
          the standard deviations of the training data.
    """
    du.util._check_kwargs(kwargs,['nonlins','outfn','means','stdevs'])
    means=kwargs.get('means',None); stdevs=kwargs.get('stdevs',None)
    super().__init__(means = means, stdevs = stdevs)
    nonlins = kwargs.get('nonlins', tuple([nn.ReLU()]))
    self.outfn = kwargs.get('outfn', None)
    assert (isinstance(widths, tuple) or isinstance(widths, list)), dedent("""\
        widths should be a tuple or a list, not a {}
    """.format(type(widths)))
    assert (isinstance(nonlins,tuple) or isinstance(nonlins,list)), dedent("""\
        'nonlins should be a tuple or a list. not a {}
    """.format(type(nonlins)))
    if len(nonlins) == 1:
      nonlins = tuple(list(nonlins) * len(widths))
    # assert the right type of nonlin[0] here and all entries
    # should the be a class of an instance of a class or either?
    #   NOTE: should be an instance evidently
    assert len(widths) == len(nonlins),dedent("""\
        len(nonlins) (which is {}) must be 1 or must have length
        equal to len(widths) (which is {})
    """.format(len(widths),len(nonlins)))
    widths = list(widths) + [n_outputs]
    self.model = nn.Sequential(nn.Linear(n_inputs, widths[0]))
    for layer in range(len(widths)-1):
      self.model.add_module('act'+str(layer),nonlins[layer])
      self.model.add_module(
          'lin'+str(layer+1),
          nn.Linear(widths[layer], widths[layer+1]))

  def forward(self, xss):
    """Forward pass tensor through the model.

    Args:
      xss (tensor): The features of shape `(n,m)` where
          `n` is the number of examples and `m` is the
          number of numbers in each example's features.

    Returns:
      tensor. The returned tensor has shape `(n,k)` where
          `n` is the number of examples and `k` is the num-
          ber of numbers in each example's targets.
    """
    outss = self.model(xss)
    if self.outfn: outss = self.outfn(outss)
    return outss

if __name__ == '__main__':
  import doctest
  doctest.testmod()

