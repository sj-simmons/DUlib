#!/usr/bin/env python3
"""model classes for ~convolutional~ neural nets.

\n`QUICK SIGNATURES`

  |ConvFFNet|($in_size$, $n_out$, $channels$, $widths$, $**kwargs$)
  |OneMetaCNN|($in_size$, $n_out$, $channels$, $**kwargs$)
  |TwoMetaCNN|($in_size$, $n_out$, $channels$, $width$, $**kwargs$)

  |metalayer|($channels$, $kernels$, $**kwargs$)
  |convFFhidden|($channels$, $conv_kernels$, $pool_kernels$)

The convolutional models defined here are built from metalay-
ers, where a single meta-layer consists of a 2d convolutional
layer followed by a 2d max-pooling layer, with ReLU inbetween.
Each model consists of a (composition of) metalayer(s) follow-
ed by a dense block.

The classes `OneMetaCNN` and `TwoMetaCNN` build, repectively, a one
metalayer model with two dense layers and a two metalayer model
with three dense layers. The class `ConvFFNet` generalizes those
two classes. All three classes extend `du.models.FFNet_`.

The functions `metalayer` and `convFFhidden` are helper funct-
ions for `ConvFFNet`.
"""
# Todo:
#   - ConvFFNet likely breaks(?) with strides and paddings other
#     than the default
#   - add options to change for example the nonlinearities.
#   - check stuff in init of classes with asserts.

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import du.utils
from du.models import FFNet_, denseFFhidden

__author__ = 'Scott Simmons'
__version__ = '0.9'
__status__ = 'Development'
__date__ = '01/21/20'
__copyright__ = """
  Copyright 2019-2020 Scott Simmons

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

def metalayer(channels, kernels, **kwargs):
  """A metalayer for a convolutional network.

  This returns a ~convolutional metalayer~ consisting of a single
  convolutional layer, followed by a nonlinearity, followed by
  a single max-pooling layer.

  Let the input to the meta-layer returned by this function be
  a tensor with shape defined by `(W_in, H_in)`, and let `(W_out,`
  `H_out)` be the shape of the resulting output tensor. Then the
  default `strides` and `paddings` lead to the following.

  If `kernels[0]`, which is the size of the square convolutional
  kernel, is odd, then the convolutional layer does not modify
  size (since by default the padding is (`kernels[0]`-1)/2 and
  the stride is 1). Meanwhile the pooling layer has (default)
  padding 0 and stride `kernels[1]`; hence it reduces both the
  width and the height by a factor of `kernels[1]`. We have:

  !Case 1!: `kernels[0]` is odd

  If `W_in` and `H_in` are both divisible by `kernels[1]`, then

                `W_out = W_in / kernels[1]`, and
                `H_out = H_in / kernels[1]`.

  >>> `ml, out_size = metalayer(channels=(1,16), kernels=(5,2))`
  >>> `ml`
  Sequential(
    (0): Conv2d(1, 16, k...=(5, 5), st...=(1, 1), pa...=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, ...)
  )
  >>> `ml(torch.rand(1, 1, 48, 64)).size()`
  torch.Size([1, 16, 24, 32])
  >>> `out_size(48, 64)`
  (24, 32)

  If one or both of `W_in` and `H_in` are not divisible by `kernels`
  `[1]`, then
              `W_out = floor(W_in/kernels[1])`, and
              `H_out = floor(H_in/kernels[1])`.

  >>> `ml, out_size = metalayer(channels=(1,16), kernels=(5,3))`
  >>> `ml(torch.rand(1, 1, 48, 64)).size()`
  torch.Size([1, 16, 16, 21])
  >>> `out_size(48, 64)`
  (16, 21)

  !Case 2! `kernels[1]` is even:

  If this case, the width and the height of data both grow by 1
  in moving through the convolution layer; hence

            `W_out = floor((W_in + 1)/kernels[1])`, and
            `H_out = floor((H_in + 1)/kernels[1])`.

  >>> `ml, out_size = metalayer(channels=(1,16),kernels=(7,2))`
  >>> `ml(torch.rand(1, 1, 48, 64)).size()`
  torch.Size([1, 16, 24, 32])
  >>> `out_size(48, 64)`
  (24, 32)

  Therefore, in any case that assumes the default `strides` and
  `paddings`, we have

            `W_out = floor((W_in + 1)/kernels[1])`, and
            `H_out = floor((H_in + 1)/kernels[1])`.

  #(Here we have excluded the case `kernels[1]` = 1 since, then,
  #the pooling layer has no effect.)

  Args:
    $channels$ (`Tuple[int]`): This tuple is interpreted as `(in_`
        `channels, out_channels)` where `in_channels` and `out_`
        `channels` are those for the convolutional layer.
    $kernels$ (`Tuple[int]`): The first integer determines the
        width and height of the convolutional kernel; the sec-
        ond, the same for the max-pooling kernel.

  Kwargs:
    $nonlin$(`nn.Module`): The nonlinearity.
    $strides$ (`Tuple[int]`): The first int is the stride of the
        convolutional layer; the second is that of the pooling
        layer. Default: `(1,kernels[1])`.
    $paddings$ (`Tuple[int]`): The first int is the padding for the
        convolutional layer; the second is that for the pooling
        layer. Default: `(int(kernels[0]/2),0)`.

  Returns:
    `(nn.Sequential, function)`. The metalayer tupled with a fun-
        tion that mapps `W_in, H_in` to `W_out, H_out`.

  """
  # this is metalayer
  du.utils._check_kwargs(kwargs,['strides','paddings'])
  nonlin = kwargs.get('nonlin',nn.ReLU())
  strides = kwargs.get('strides',(1,kernels[1]))
  paddings = kwargs.get('paddings',(int(kernels[0]/2),0))
  ml = nn.Sequential(
           nn.Conv2d(
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
  def out_size(width, height):
    return int((width+1)/kernels[1]), int((height+1)/kernels[1])
  return ml, out_size

def convFFhidden(channels, conv_kernels, pool_kernels):
  """Compose convolutional metalayers.

  This composes the specified convolutional metalaters into a
  block for use in the hidden part a feed-forward neural net.
  Let `n` denote the number of specified metalayers; that is,
  `n = len(conv_kernels) = len(pool_kernels) = len(channels)-1`.

  Args:
    $channels$ (`Tuple[int]`): A tuple of length `n+1` the first ent-
        ry of which is `in_channels` for the first metalayer's
        convolutional part; the rest of the entries are the su-
        ccessive `out_channels` for the convolutional part of the
        first meta-layer, the second meta-layer, etc.
    $conv_kernels$ (`Tuple[int]`): A tuple of length `n` holding the
        kernel size for the convolution part the successive
        metalayer.
    $pool_kernels$ (`Tuple[int]`): A tuple of length `n` holding
        the kernel size for the pooling layer of successive
        metalayer.

  Returns:
    `(nn.Sequential, function)`. The block consisting of the com-
        posed metalayers tupled with a function mapping `W_in,`
        `H_in` to `W_out, H_out` where `(W_in, H_in)` is the shape of
        an input to the bock and `(W_out, H_out)` is the corres-
        ponding output.

  >>> `convFFhidden((1,32, 64), (5,3), (2,2))`
  (Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(5, 5), ...)
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, ...)
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), ...)
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, ...)
    )
  ), ...)
  """
  assert len(channels)-1 == len(conv_kernels) == len(pool_kernels)
  layers, funcs = list(zip(*[metalayer(chans, kerns) for chans, kerns in zip(
      zip(channels[:-1],channels[1:]), zip(conv_kernels, pool_kernels))]))
  return nn.Sequential(*layers), functools.reduce(
      lambda f,g: lambda x,y:g(*f(x,y)), funcs, lambda x,y:(x,y))

class ConvFFNet(FFNet_):
  """Meta-layered convolutional net.

  Builds a convolutional net consisting of the composition of
  convolutional metalayers followed by dense layers.
  """
  def __init__(self, in_size, n_out, channels, widths, **kwargs):
    """Constructor.

    Args:
      $in_size$ (`Tuple[int]`): A tuple of length 2 holding the
          width and height of each input.
      $n_out$ (`int`): Number of outputs from the model in its
          entirety. This would be 10 to say classify digits,
          or 1 for a regression problem.
      $channels$ (`Tuple[int]`): The first entry set `in_channe`
          `ls` for the first metalayer's convolutional part;
          the rest of the entries are the successive `out_cha`
          `nnels` for the convolutional part of the first met-
          alayer, the second metalayer, etc.
      $widths$ (`Tuple[int]`): The widths (no. of nodes) in the
          successive layers of the dense part.

    Kwargs:
      $conv_kernels$ (`Tuple[int]`): Default: `(len(channels)-1)*[5]`
      $pool_kernels$ (`Tuple[int]`): Default: `(len(channels)-1)*[2]`
      $outfn$ (`nn.Module`): a function to pipe out though lastly
          in the `forward` method; The default is `log_softmax`.
          For regression, you likely want to put `None`.
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data.
      $stdevs$ (`torch.Tensor`): A tensor typically holding the
          standard deviations of the training data.
    """
    du.utils._check_kwargs(kwargs, ['conv_kernels','pool_kernels','means',
        'stdevs','outfn'])
    means = kwargs.get('means', None)
    stdevs = kwargs.get('stdevs', None)
    assert len(in_size) == 2,\
        'in_size must have length 2 not {}'.format(len(in_size))
    super().__init__(means = means, stdevs = stdevs)
    self.outfn = kwargs.get('outfn',
        lambda xss: torch.log_softmax(xss,dim=1))
    conv_kernels = kwargs.get('conv_kernels',(len(channels)-1)*[5])
    pool_kernels = kwargs.get('pool_kernels',(len(channels)-1)*[2])

    # build the convolutional part:
    self.conv, out_size = convFFhidden(channels, conv_kernels, pool_kernels)

    # build the dense part
    self.dense = denseFFhidden(
        n_inputs = channels[-1]*(lambda x,y: x*y)(*out_size(*in_size)),
        n_outputs = n_out,
        widths = widths)

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    the convolutional part of the model followed by the ful-
    ly-connected part.

    Args:
      $xss$ (`torch.Tensor`): The tensor to be forwarded.

    Returns:
      (`torch.Tensor`). The forwarded tensor.
    """
    xss = self.conv(xss.unsqueeze(1))
    xss = self.dense(xss.reshape(len(xss),-1))
    if self.outfn: xss = self.outfn(xss)
    return xss

class OneMetaCNN(FFNet_):
  """One meta-layer CNN with a two fully-connected layers.

  Note: Consider using `DenseFFNet` which generalizes this.
  """
  def __init__(self, in_size, n_out, channels, **kwargs):
    """Constructor.

    Args:
      $in_size$ (`Tuple[int]`): A tuple of length 2 holding the
          width and height of each input.
      $n_out$ (`int`): Number of outputs from the model. This is
          10 to classify digits, or 1 for a regression problem.
      $channels$ (`Tuple(int)`). This is `(in_channels, out_chann`
          `els)` where 'channels' is that of the convolutional
          part of the metalayer.
    Kwargs:
      $outfn$ (`nn.Module`): a function to pipe out though lastly
          in the `forward` method; The default is `log_softmax`.
          For regression, you likely want to put `None`.
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data.
      $stdevs$ (`torch.Tensor`): A tensor typically holding the
          standard deviations of the training data.
    """
    du.utils._check_kwargs(kwargs, ['means', 'stdevs', 'outfn'])
    means = kwargs.get('means', None)
    stdevs = kwargs.get('stdevs', None)
    self.outfn = kwargs.get('outfn',
        lambda xss: torch.log_softmax(xss,dim=1))
    assert len(in_size) == 2,\
        'in_size must have length 2 not {}'.format(len(in_size))
    super().__init__(means = means, stdevs = stdevs)
    self.meta_layer = nn.Sequential(
        nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size = 5,
            stride = 1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )
    self.fc_layer = nn.Linear(int(channels[1]*in_size[0]*in_size[1]/4),n_out)

    self.register_buffer('means', means)
    self.register_buffer('stdevs', stdevs)

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    in turn, a meta-layer and a fully-connected layer.

    Args:
      $xss$ (`torch.Tensor`): The tensor to be forwarded.

    Returns:
      (`torch.Tensor`). The forwarded tensor.
    """
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer(xss)
    xss = self.fc_layer(xss.reshape(len(xss),-1))
    if self.outfn: xss = self.outfn(xss)
    return xss

class TwoMetaCNN(FFNet_):
  """Two meta-layer CNN with three fully-connected layers.

  Note: Consider using `DenseFFNet` which generalizes this.
  """
  def __init__(self, in_size, n_out, channels, width, **kwargs):
    """Constructor.

    Args:
      $in_size$ (`Tuple[int]`): A tuple of length 2 holding the
          width and height of each input.
      $n_out$ (`int`): Number of outputs from the model. This is
          10 to classify digits, or 1 for a regression problem.
      $channels$ (`Tuple(int)`). This is a triple the first ent-
          ry of which is `in_channels` for the convolutional
          part of the first metalayer; the second and third
          entries are `out_channels` for the convolutional
          parts of the first and second metalayers, resp.
      $width$ (`int`): the widths (no. of nodes) in the second
          layers of the dense part.

    Kwargs:
      $outfn$ (`nn.Module`): a function to pipe out though lastly
          in the `forward` method; The default is `log_softmax`.
          For regression, you likely want to put `None`.
      $means$ (`torch.Tensor`): A tensor typically holding the
          means of the training data.
      $stdevs$ (`torch.Tensor`): A tensor typically holding the
          standard deviations of the training data.
    """
    du.utils._check_kwargs(kwargs, ['means', 'stdevs', 'outfn'])
    means = kwargs.get('means', None)
    stdevs = kwargs.get('stdevs', None)
    assert len(in_size) == 2,\
        'in_size must have length 2 not {}'.format(len(in_size))
    self.outfn = kwargs.get('outfn',
        lambda xss: torch.log_softmax(xss,dim=1))
    super().__init__(means = means, stdevs = stdevs)
    self.metalayer1 = nn.Sequential(# A mini-batch of size of N to this should
                                     # have size:
        nn.Conv2d(                   # N x channels[0] x in_size[0] x in_size[1]
            in_channels=channels[0],
            out_channels=channels[1],# And the output of Conv2d is still size:
            kernel_size=5,           # N x channels[1] x in_size[0] x in_size[1]
            stride=1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )                                 # Downsampling with MaxPool we have that
    self.metalayer2 = nn.Sequential( # the input here is:
        nn.Conv2d(                    # N x channels[1] x 10 x 10.
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,      # And the ouput of this Conv2d is:
            stride=1,           # N x channels[2] x in_size[0]/2 x in_size[1]/2.
            padding = 1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )                           # Downsampling, again, we have
                                # N x channels[2] x in_size[0]/4 x in_size[1]/4.
    self.fc_layer1 = nn.Linear(int(channels[2]*in_size[0]*in_size[1]/16), width)
    self.fc_layer2 = nn.Linear(width, n_out)

  def forward(self, xss):
    """Forward inputs.

    Forwards features (of a mini-batch of examples) through,
    in turn, two meta-layers and two fully-connected layers,
    followed by logsoftmax.

    Args:
      $xss$ (`torch.Tensor`): The tensor to be forwarded.

    Returns:
      (`torch.Tensor`). The forwarded tensor.
    """
    xss = self.metalayer2(self.metalayer1(xss.unsqueeze(1)))
    xss = self.fc_layer1(xss.reshape(len(xss),-1))
    xss = self.fc_layer2(torch.relu(xss))
    return torch.log_softmax(xss, dim=1)

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
