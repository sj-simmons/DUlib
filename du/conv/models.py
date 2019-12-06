#!/usr/bin/env python3
'''model classes for ~convolutional neural nets~.

The convolutional models defined here are built from metalay-
ers, where a single meta-layer consists of a 2d convolution
layer followed by a 2d max-pooling layer.

Currently, two convolutional models are defined. The signatur-
es for their constructors are as follows. See their class doc-
umentation below for more.

OneMetaCNN
  (means: tensor = None, stdevs: tensor = None)

TwoMetaCNN
  (width_m1:int=16, width_m2:int=32, means=None, stdevs=None)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from du.models import FFNNet_

__author__ = 'Scott Simmons'
__version__ = '0.8.5'
__status__ = 'Development'
__date__ = '12/06/19'

#def MetaLayer(nn.Module):
#  """A metalayer for a convolutional network.
#
#  This is a customizable single so-called 'meta-layer' con-
#  sisting of a single convolutional layer followed by a sin-
#  gle max-pooling layer.
#
#  Args:
#    in_size (Tuple[int]): A tuple of length two holding the
#        width and length of each input. If an instance of
#        this class is the first meta-layer encounterd by data
#        on its way through a convolutional network, then this
#        is just `(width, length)` where `width` and `length`
#        are the feature image width and length in pixels for
#        an example in the data.
#    channels (Tuple[int]): `(in_channels, out_channels)`.
#    kernel_size (int): The width and height of the kernel.
#    stride (int)
#    padding (int)
#  """
#  def __init__(self, im_size, in_width, kernel_size):
#  pass


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
    '''Forwards through, in turn, a single meta-layer and a
    single fully-connect layer, followed by logsoftmax.
    '''
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
    '''Forwards examples through, in turn, two meta-layers and
    two fully-connected layers, followed by logsoftmax.
    '''
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
  failures, _ = doctest.testmod()

  if failures == 0:
    # Below prints only the signature of locally defined functions.
    from inspect import signature
    local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
        if callable(ob) and ob.__module__ == __name__]
    for name, ob in local_functions:
      print(name,'\n  ',signature(ob))
