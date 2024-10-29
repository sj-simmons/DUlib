#!/usr/bin/env python3
"""model classes for ~convolutional~ neural nets.

\n`QUICK SIGNATURES`

  |ConvFFNet|($in_size$, $n_out$, $channels$, $widths$, $**kwargs$)
  |OneMetaCNN|($in_size$, $n_out$, $channels$, $**kwargs$)
  |TwoMetaCNN|($in_size$, $n_out$, $channels$, $width$, $**kwargs$)

  |metalayer|($channels$, $kernels$, $nonlin$, $**kwargs$)
  |convFFhidden|($channels$, $conv_kernels$, $pool_kernels$,$**kwargs$)

The convolutional models defined here are built from metalay-
ers, where a single meta-layer consists of a 2d convolutional
layer followed by a 2d max-pooling layer, with ReLU inbetween.
Each model consists of a (composition of) metalayer(s) follow-
ed by a dense block.

The classes `OneMetaCNN` and `TwoMetaCNN` build, repectively, a one
metalayer model with two dense layers and a two metalayer model
with three dense layers. The class `ConvFFNet` generalizes those
two classes. All three classes extend `du.models.FFNet_`.

The functions `metalayer` and `convFFhidden` are helper functions
for `ConvFFNet`.
"""
# Todo:
#   - ConvFFNet and others(?) only works with 1 channel input
#     to the first conv layer (so only b&w images??).
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

def metalayer(channels, kernels, nonlin, **kwargs):
  """A metalayer for a convolutional network.

  This returns a ~convolutional metalayer~ consisting of a single
  convolutional layer, followed by a nonlinearity, followed by
  a single max-pooling layer.

  Let the input to the meta-layer returned by this function be
  a tensor with shape defined by `(H_in, W_in)`, and let `(H_out,`
  `W_out)` be the shape of the resulting output tensor. Then the
  default `strides` and `paddings` lead to the following.

  If `kernels[0]`, which is the size of the square convolutional
  kernel, is odd, then the convolutional layer does not modify
  size (since by default the padding is (`kernels[0]`-1)/2 and
  the stride is 1). Meanwhile the pooling layer has (default)
  padding 0 and stride `kernels[1]`; hence it reduces both the
  height and the width by a factor of `kernels[1]` if `kernels[1]`
  is divides both height and width. More generally, We have:

  !Case 1!: `kernels[0]` is odd

  If `H_in` and `W_in` are both divisible by `kernels[1]`, then

                `H_out = H_in / kernels[1]`, and
                `W_out = W_in / kernels[1]`.

  >>> `ml, out_size = metalayer((1,16), (5,2), nn.ReLU())`
  >>> `ml`
  Sequential(
    (0): Conv2d(1, 16, k...=(5, 5), st...=(1, 1), pa...=(2, 2))
    (1): BatchNorm2d(16, ...)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, ...)
  )
  >>> `ml(torch.rand(1, 1, 48, 64)).size()`
  torch.Size([1, 16, 24, 32])
  >>> `out_size(48, 64)`
  (24, 32)

  If one or both of `H_in` and `W_in` are not divisible by `kernels`
  `[1]`, then
              `H_out = floor(H_in/kernels[1])`, and
              `W_out = floor(W_in/kernels[1])`.

  >>> `ml, out_size = metalayer((1,16), (5,3), nn.ReLU())`
  >>> `ml(torch.rand(1, 1, 48, 64)).size()`
  torch.Size([1, 16, 16, 21])
  >>> `out_size(48, 64)`
  (16, 21)
  >>> `ml, out_size = metalayer((1,16), (5,2), nn.ReLU())`
  >>> `ml(torch.rand(1, 1, 47, 64)).size()`
  torch.Size([1, 16, 23, 32])
  >>> `out_size(47, 64)`
  (23, 32)

  !Case 2! `kernels[0]` is even:

  If this case, the height and the width of data both grow by 1
  in moving through the convolution layer; hence

            `H_out = floor((H_in + 1)/kernels[1])`, and
            `W_out = floor((W_in + 1)/kernels[1])`.

  >>> `ml, out_size = metalayer((1,16), (6,2), nn.ReLU())`
  >>> `ml(torch.rand(1, 1, 47, 64)).size()`
  torch.Size([1, 16, 24, 32])
  >>> `out_size(47, 64)`
  (24, 32)

  Therefore, in any case that assumes the default `strides` and
  `paddings`, we have

  `H_out = floor((H_in + (kernel[0]+1) mod 2))/kernels[1])`, and
  `W_out = floor((W_in + (kernel[0]+1) mod 2))/kernels[1])`.

  (Here we have excluded the case `kernels[1]` = 1 since, then,
  the pooling layer has no effect.)

  >>> `ml, out_size = metalayer((1,16), (7,2), nn.ReLU())`
  >>> `ml(torch.rand(1, 1, 47, 64)).size()`
  torch.Size([1, 16, 23, 32])
  >>> `out_size(47, 64)`
  (23, 32)

  >>> `ml, out_size = metalayer((1,16), (7,3), nn.ReLU())`
  >>> `ml(torch.rand(1, 1, 47, 64)).size()`
  torch.Size([1, 16, 15, 21])
  >>> `out_size(47, 64)`
  (15, 21)

  Args:
    $channels$ (`Tuple[int]`): This tuple is interpreted as `(in_`
        `channels, out_channels)` where `in_channels` and `out_`
        `channels` are those for the convolutional layer.
    $kernels$ (`Tuple[int]`): The first integer determines the
        width and height of the convolutional kernel; the sec-
        ond, the same for the max-pooling kernel.
    $nonlin$ (`nn.Module`): The nonlinearity.

  Kwargs:
    $strides$ (`Tuple[int]`): The first int is the stride of the
        convolutional layer; the second is that of the pooling
        layer. Default: `(1, kernels[1])`.
    $paddings$ (`Tuple[int]`): The first int is the padding for the
        convolutional layer; the second is that for the pooling
        layer. Default: `(int(kernels[0]/2), 0)`.
    $batchnorm$ ('(str, kwargs)'): A tuple which, if not emp-
        ty, results in a batch normalization layer being inser-
        ted in the metalayer. If the string in the first posit-
        ion is 'before', respectively 'after', then batch nor-
        malization takes place before, resp. after, applying
        the nonlinearity. Keyword arguments for `torch.nn.Batch`
        `Norm2d` can also be supplied in the form of a `dict`.
        Default: `('before',)`.
    $dropout$ (`float`): If greater than zero, add a dropout layer
        with this probablity before each nonlinearity.  Def: `0`.

  >>> `bn = ('before',{'momentum':.99})
  >>> `ml,_= metalayer((1,16), (7,3), nn.ReLU(), batchnorm=bn)`
  >>> `ml(torch.rand(1, 1, 47, 64)).size()`
  torch.Size([1, 16, 15, 21])

  Returns:
    `(nn.Sequential, function)`. The metalayer tupled with a fun-
        tion that mapps `H_in, W_in` to `H_out, W_out`.
  """
  # this is metalayer
  du.utils._check_kwargs(kwargs,['strides','paddings','batchnorm','dropout'])
  strides = kwargs.get('strides',(1,kernels[1]))
  paddings = kwargs.get('paddings',(int(kernels[0]/2),0))
  batchnorm = kwargs.get('batchnorm', ('before',))
  dropout = kwargs.get('dropout', 0)
  if dropout > 0:
    if batchnorm:
      if len(batchnorm) == 1: bn_kwargs = {} # batchnorm kwargs
      else:
        bn_kwargs = batchnorm[1]
        assert isinstance(bn_kwargs,dict),\
           'second element of batchnorm must be a dict'
      if kernels[1] > 1:
        ml=nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(num_features=channels[1], **bn_kwargs),
            nn.Dropout(dropout),
            nonlin,
            nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1],
            padding=paddings[1]))
      else:
        ml = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(num_features=channels[1], **bn_kwargs),
            nonlin)
    else:
      if kernels[1] > 1:
        ml=nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.Dropout(dropout),
            nonlin,
            nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1],
            padding=paddings[1]))
      else:
        ml = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nonlin)
  else:
    if batchnorm:
      if len(batchnorm) == 1: bn_kwargs = {} # batchnorm kwargs
      else:
        bn_kwargs = batchnorm[1]
        assert isinstance(bn_kwargs,dict),\
           'second element of batchnorm must be a dict'
      if kernels[1] > 1:
        ml=nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(num_features=channels[1], **bn_kwargs),
            nonlin,
            nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1],
            padding=paddings[1]))
      else:
        ml = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(num_features=channels[1], **bn_kwargs),
            nonlin)
    else:
      if kernels[1] > 1:
        ml=nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nonlin,
            nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1],
            padding=paddings[1]))
      else:
        ml = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nonlin)
  def out_size(height, width):
    return tuple(ml(torch.randn(1,channels[0],height,width)).size()[2:])
    #return int((height + (kernels[0] + 1) % 2) / kernels[1]),\
    #       int((width + (kernels[0] + 1) % 2) / kernels[1])
  return ml, out_size

def convFFhidden(channels, conv_kernels, pool_kernels, **kwargs):
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

  Kwargs:
    $nonlins$ (`nn.Module`): The nonlinearities to compose bet-
        ween meta-layers. Default: `nn.ReLU()`.
    $batchnorm$ (`(str, kwargs)`): A tuple which, if not empty, re-
        sults in a batch normalization layer being inserted in
        each convolutional metalayer. If the string in the
        first position is 'before', resp. 'after', then batch
        normalization takes place before, resp.  after, the
        nonlinearity in each convolutional metalayer. Keywords
        for `torch.nn.BatchNorm2d` can be supplied in the form a
        `dict` and included as the second element of this tuple;
        those will be applied in each convolutional metalayer's
        batch normalization layer. Default: `('before',)`.
    $dropout$ (`float`): If greater than zero, add a dropout layer
        with this probablity before each nonlinearity.  Def: `0`.

  Returns:
    `(nn.Sequential, function)`. The block consisting of the com-
        posed metalayers tupled with a function mapping `W_in,`
        `H_in` to `W_out, H_out` where `(W_in, H_in)` is the shape of
        an input to the block and `(W_out, H_out)` is the corres-
        ponding output.

  >>> `convFFhidden((1,32, 64), (5,3), (2,2), batchnorm=())`
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
  du.utils._check_kwargs(kwargs,['nonlins','batchnorm','dropout'])
  nonlins = kwargs.get('nonlin',nn.ReLU())
  dropout = kwargs.get('dropout', 0)
  batchnorm = kwargs.get('batchnorm', ('before',))
  assert len(channels)-1 == len(conv_kernels) == len(pool_kernels)
  layers,funcs=list(
      zip(*[metalayer(chans,kerns,nonlins,batchnorm=batchnorm,dropout=dropout)
          for chans, kerns in zip(
              zip(channels[:-1],channels[1:]),
                  zip(conv_kernels, pool_kernels))]))
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
          $in_size$ (`Tuple[int]`): A tuple (height, width) holding
              the height and width of each input (in pixels, for
              images).
          $n_out$ (`int`): Number of outputs from the model in its
              entirety. This would be 10 to say classify digits,
              or 1 for a regression problem.
          $channels$ (`Tuple[int]`): The first entry sets `in_channels`
              for the first metalayer's convolutional part; the
              rest of the entries are the successive `out_chann`
              `els` for the convolutional part of the first meta-
              layer, the second metalayer, etc.
          $widths$ (`Tuple[int]`): The widths (no. of nodes) in the
              successive layers of the dense part.

        Kwargs:
          $conv_kernels$ (`Tuple[int]`): Default: `(len(channels)-1)*[5]`
          $pool_kernels$ (`Tuple[int]`): Default: `(len(channels)-1)*[2]`
          $nonlins$ (`Tuple[nn.Module]`): A length 2 tuple determin-
              ing the nonlinearities for, resp., the convolution-
              al and the dense parts of the network. Default: `(nn`
              `.ReLU(), nn.ReLU())`.
          $batchnorm$ (`(str, kwargs)`): A tuple which, if not empty,
              results in a batch normalization layer being inser-
              ted in each convolutional metalayer. If the string
              in the first position is 'before', resp. 'after',
              then batch normalization takes place before, resp.
              after, the nonlinearity in each convolutional meta-
              layer. Keywords for `torch.nn.BatchNorm2d` can be
              supplied in the form a `dict` and included as the
              second element of this tuple; those will be applied
              in each convolutional metalayer's batch normalizat-
              ion layer. Default: `('before',)`.
          $dropout$ (`Tuple(float)`): In the case that this tuple
              has length 2: if the first (last) float is greater
              than zero, add dropout with this probablity to each
              layer of the convolutional (dense) part before the
              nonlinearity.
              If this tuple the length 3, then the first entry
              determines dropout probability for the input layer;
              while the last 2 numbers function as above.
              Default: `(0, 0, 0)`.
          $outfn$ (`nn.Module`): A function to pipe out though lastly
              in the `forward` method; The default is `log_softmax`.
              For regression, you likely want to put `None`.
          $means$ (`torch.Tensor`): A tensor typically holding the
              means of the training data.
          $stdevs$ (`torch.Tensor`): A tensor typically holding the
              standard deviations of the training data.

        >>> `model = ConvFFNet((28,28), 10, (1,16,8), (100,50))`
        >>> `xss = torch.rand(100,28,28)` # e.g., b&w images
        >>> `yhatss = model(xss)`
        >>> `yhatss.size()`
        torch.Size([100, 10])

        >>> `bn = ('before',{'momentum':0.9})`
        >>> `model=ConvFFNet((28,28),8,(1,16),(100,),batchnorm=bn)`
        >>> `xss = torch.rand(100,28,28)` # e.g., b&w images
        >>> `yhatss = model(xss)`
        >>> `yhatss.size()`
        torch.Size([100, 8])

        >>> `print(model.short_repr(color=False))`
        Conv.: channels 1 16 ReLU batchnorm:before dropout:0
        Dense: widths 3136 100 8 ReLU dropout:0

        >>> `model=ConvFFNet((28,28),8,(1,16),(),batchnorm=bn)`
        >>> `print(model.short_repr(color=False))`
        Conv.: channels 1 16 ReLU batchnorm:before dropout:0
        Dense: widths 3136 8 ReLU dropout:0
        """
        du.utils._check_kwargs(
             kwargs,
             ['conv_kernels','pool_kernels','means', 'stdevs','outfn','nonlins','batchnorm','dropout'])
        means = kwargs.get('means', None)
        stdevs = kwargs.get('stdevs', None)
        super().__init__(means = means, stdevs = stdevs)
        self.outfn = kwargs.get('outfn', lambda xss: torch.log_softmax(xss,dim=1))
        conv_kernels = kwargs.get('conv_kernels',(len(channels)-1)*[5])
        pool_kernels = kwargs.get('pool_kernels',(len(channels)-1)*[2])
        nonlins = kwargs.get('nonlins', (nn.ReLU(), nn.ReLU()))
        dropout = kwargs.get('dropout', (0, 0))
        batchnorm = kwargs.get('batchnorm', ('before',))

        assert len(in_size) == 2, \
            du.utils._markup(f'The augument to `in_size` must have length 2 not {len(in_size)}')
        assert 2 <=len(dropout) <=3, \
            du.utils._markup(f'The argument to `dropout` must have length 2 or 3 not {len(dropout)}')
        self.dropfeats = dropout[0] if len(dropout) == 3 else 0
        dropout = dropout[-2:]

        # build the convolutional part:
        self.conv, out_size = convFFhidden(
            channels, conv_kernels, pool_kernels, nonlins = nonlins[0],
            batchnorm = batchnorm, dropout = dropout[0])
        # build the dense part
        n_inputs_dense = channels[-1]*(lambda x,y: x*y)(*out_size(*in_size))
        self.dense = denseFFhidden(
            n_inputs = n_inputs_dense, n_outputs = n_out, widths = widths,
            nonlins = (nonlins[1],), dropout = dropout[1])

        # build a short representation string
        nonlins = list(map(lambda mo: repr(mo)[repr(mo).rfind('.')+1:-2], nonlins))
        batchnorm = 'none' if len(batchnorm)==0 else batchnorm[0]
        inputpart = f'Inputs: ~dropout~:`{self.dropfeats}`\n' if self.dropfeats else ''
        convpart = functools.reduce(
            lambda x, y: x + ' ' + y,
            ['Conv.: ~channels~']+list(map(lambda x: '`'+str(x)+'`',channels))+['`'+nonlins[0]+'`']\
            + ['~batchnorm~:'+ '`'+str(batchnorm)+'`'] + ['~dropout~:'+'`'+str(dropout[0])+'`'])
        densepart = functools.reduce(
            lambda x, y: x + ' ' + y,
            ['\nDense: ~widths~'] \
            + list(map(lambda x: '`'+str(x)+'`', (n_inputs_dense,) + tuple(widths) + (n_out,)))\
            + ['`'+nonlins[1]+'`'] + ['~dropout~:'+'`'+str(dropout[1])+'`'])
        self.repr_ = inputpart + convpart + densepart

    def forward(self, xss):
        """Forward inputs.

        Forwards features (of a mini-batch of examples) through
        the convolutional part of the model followed by the ful-
        ly-connected part.

        Args:
          $xss$ (`Tensor`): The tensor to be forwarded.

        Returns:
          (`Tensor`). The forwarded tensor.
        """
        xss = self.conv(xss.unsqueeze(1))
        if self.dropfeats:
            xss = nn.Dropout(self.dropfeats)(xss)
        xss = self.dense(xss.reshape(len(xss),-1))
        if self.outfn:
            xss = self.outfn(xss)
        return xss

    def short_repr(self, color=True):
        """Return concise representaton string."""
        return du.utils._markup(self.repr_, strip = not color)

class OneMetaCNN(FFNet_):
  """One meta-layer CNN with a two fully-connected layers.

  Note: Consider using `ConvFFNet` which generalizes this.
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

  Note: Consider using `ConvFFNet` which generalizes this.
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
          layer of the dense part.

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
