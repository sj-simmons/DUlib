#!/usr/bin/env python3
"""model classes for ~convolutional neural nets~.

The convolutional models defined here are built from meta-
layers, where a single meta-layer consists of a 2d convol-
utional layer followed by a 2d max-pooling layer.

The classes `OneMetaCNN` and `TwoMetaCNN` build, repective-
ly, one and two meta-layer models. The class `ConvFFNet`
(which extends `du.models.FFNNet_`) generalizes those two
classes.

The two functions `ConvMetaLayer` and `ConvFFHidden` are
helper functions for `ConvFFNet`.
"""
# Todo:
#   - generalize OneMetaCNN, and then add kwargs to it and
#     to TwoMetaCNN

from collections import OrderedDict
from fractions import Fraction
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import du.util
from du.models import FFNNet_

__author__ = 'Scott Simmons'
__version__ = '0.9'
__status__ = 'Development'
__date__ = '12/16/19'

def ConvMetaLayer(channels, kernels, **kwargs):
  """A metalayer for a convolutional network.

  This returns the pieces, ready to be composed, of a meta-
  -layer consisting of a single convolutional layer, follow-
  ed by a nonlinearity, followed by a single max-pooling
  layer.

  Let the input to this layer have size `W_in` by `H_in`,
  and let the output have size `W_out` by `H_out`.  Then the
  default `strides` and `paddings` lead to the following
  relationships between input and output sizes:

  `W_out
  


  If kernels[0], which is the size of the square convolu-
  tional kernel, is even then the convolutional layer does
  not modify the size (since by default

  Args:
    channels (Tuple[int]): This tuple is interpreted as
        `(in_channels, out_channels)` where `in_channels`
        and `out_channels` are that of the convolutional
        layer.
    kernels(Tuple[int]): The first integer determines the
        width and height convolutional kernel; the second,
        the max-pooling kernel.

  Kwargs:
    nonlin (nn.Module): The nonlinearity
    strides (Tuple[int]): The first int is the stride of
        the convolutional layer; the second is that of the
        pooling layer. Default: `(1,kernels[1])`.
    paddings (Tuple[int]): The first int is the padding for
        the convolutional layer; the second is that for the
        pooling layer. Default: `(int(kernels[0]/2),0)`.

  Returns:
    Tuple(nn.Module). A tuple whose items are the convolu-
        tional layer, the nonlinearity, and the pooling
        layer.

  >>> du.conv.models.ConvMetaLayer((1,16),(5,2))
  (Conv2d(1, 16,...), ReLU(), MaxPool2d(kernel_size=2,...))
  """
  du.util._check_kwargs(kwargs,['strides','paddings'])
  nonlin = kwargs.get('nonlin',nn.ReLU())
  strides = kwargs.get('strides',(1,kernels[1]))
  paddings = kwargs.get('paddings',(int(kernels[0]/2),0))
  return (nn.Conv2d(
              in_channels = channels[0],
              out_channels = channels[1],
              kernel_size = kernels[0],
              stride = strides[0],
              padding = paddings[0]),
          nonlin,
          nn.MaxPool2d(
              kernel_size = kernels[1],
          stride = strides[1],
          padding = paddings[1]))

def ConvFFHidden(channels, conv_kernels, pool_kernels):
  """A composition of convolutional meta-layers.

  Returns the feed-forward composition of `n` convolutional
  meta-layers.

  Args:
    channels (Tuple[int]): A tuple of length `n+1` the first
        entry of which is `in_channels` for the first meta-
        layer's convolutional part; the rest of the entries
        are the successive out_channels for the convolutional
        part of the first meta-layer, the second, etc.
    conv_kernels (Tuple[int]): A tuple of length `n` holding
        the kernel size for the convolution part successive
        meta-layer.
    pool_kernels (Tuple[int]): A tuple of length `n` holding
        the kernel size for the pooling layer of successive
        meta-layer.

  Returns:
    OrderedDict.

  >>> ConvFFHidden((1,16), (5,), (2,))
  Ord...('conv0', Conv2d...), ('relu', ReLU()), ('maxpool0'...)
  >>> du.conv.models.ConvFFHidden((1,16,32), (5,5), (2,2))
  OrderedDict([('conv0'...),..., ('conv1'...), ...])
  """
  assert len(channels)-1 == len(conv_kernels) == len(pool_kernels)
  chan_tups = [(channels[i], channels[i+1]) for i in range(len(channels)-1)]
  d = OrderedDict()
  for i, (chans, kerns) in\
      enumerate(zip(chan_tups, zip(conv_kernels, pool_kernels))):
    d['conv'+str(i)], d['relu'], d['maxpool'+str(i)]=ConvMetaLayer(chans, kerns)
  return d

class ConvFFNet(FFNNet_):
  """Meta-layered convolutional net.

  Builds a convolutional net consisting of the composition
  of meta-layers followed by dense layers.
  """
  def __init__(self, n_out, channels, widths, **kwargs):
    """Constructor.
    Args:
      in_size (Tuple[int]): A tuple of length two holding the
          width and height of each input. If an instance of
          this class is the first meta-layer encounterd by data
          on its way through a convolutional network, then this
          is just `(width, height)` where `width` and `height`
          are the feature image width and length in pixels for
          an example in the data.
      n_out (int): Number of outputs from the model in its
          entirety.
      channels (`Tuple[int]`): The widths (i.e., number of
          nodes) in the successive layers of the dense part.
      widths (`Tuple[int]`): The widths (i.e., number of
          nodes) in the successive layers of the dense part.

    Kwargs:
      outfn (`nn.Module`): a function to pipe out though
          lastly in the `forward` method; The default is
          `lambda outss: log_softmax(outss, dim=1)`. For a
          regression problem, you likely want to put `None`.

    """
    du.util._check_kwargs(kwargs,['conv_kernels','pool_kernels','means',
        'stdevs','outfn'])
    means = kwargs.get('means', None)
    stdevs = kwargs.get('stdevs', None)
    super().__init__(means = means, stdevs = stdevs)
    self.outfn = kwargs.get('outfn', lambda outss: nn.log_softmax(outss,dim=1))
    conv_kernels=skwargs.get('conv_kernels',(len(channels)-1)*[5])
    pool_kernels=skwargs.get('pool_kernels',(len(channels)-1)*[2])
    self.conv = convFFhidden(channels, conv_kernels, pool_kernels)
    # Let us compute the size of an image as it emerges from the convol'al part.
    for 

    self.dense = denseFFhidden()

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    in turn, a meta-layer and a fully-connected layer, fol-
    lowed by logsoftmax.

    Args:
      `xss` (`torch.Tensor`): The tensor to be forwarded.

    Returns:
      (`torch.Tensor`). The forwarded tensor.
    """
    outss = self.conv(xss)
    outss = self.dense(outss.reshape(len(xss),-1))
    outss = self.dense(outss)
    if self.outfn: outss = self.outfn(outss)
    return outss

class OneMetaCNN(FFNNet_):
  '''Class for a convolutional model with a single meta-layer.
  '''
  def __init__(self, means = None, stdevs = None):
    '''Constructor.

    Args:
      means (torch.Tensor): A tensor typically holding the
          means of the training data.
      stdevs (torch.Tensor): A tensor typically holding the
          standard deviations of the training data.
    '''
    super().__init__(means = means, stdevs = stdevs)
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

    # Add mean and stdev fields to state_dict.  These hold
    # the means and stdevs of the training data.
    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    in turn, a meta-layer and a fully-connected layer, fol-
    lowed by logsoftmax.
    """
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = torch.reshape(xss, (-1, 1600))
    xss = self.fc_layer1(xss)
    return torch.log_softmax(xss, dim=1)

class TwoMetaCNN(FFNNet_):
  ''' A two meta-layer convolutional model.
  '''
  def __init__(self, im_size = (20, 20), width_m1=16, width_m2=32, width_fc1 = 200, n_classes = 10, means = None, stdevs = None):
    super().__init__(means = means, stdevs = stdevs)
    '''Constructor.

    Args:
      width_m1 (int): The number of out-channels for the first
          meta-layer that is encountered by forwarded examples.
          Default: 16.
      width_m2 (int): The number of out-channels of the second
          meta-layer that is encountered by forwarded examples.
          Default: 32.
      means (torch.Tensor): A tensor typically holding the
          means of the training data. Default: None.
      stdevs (torch.Tensor): A tensor typically holding the
          standard deviations of the training data. Default: None.
    '''
    self.im_w = im_size[0]
    self.im_h = im_size[1]
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
    self.fc_layer1 = nn.Linear(int(width_m2*self.im_h*self.im_w/16), width_fc1)
    self.fc_layer2 = nn.Linear(width_fc1, n_classes)

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    in turn, two meta-layers and two fully-connected layers,
    followed by logsoftmax.
    """
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = self.meta_layer2(xss)
    xss = torch.reshape(xss, (-1, int(self.width_m2*self.im_h*self.im_w/16)))
    xss = self.fc_layer1(xss)
    xss = torch.relu(xss)
    xss = self.fc_layer2(xss)
    return torch.log_softmax(xss, dim=1)


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
