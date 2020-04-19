#!/usr/bin/env python3
"""core functionality for working with neural nets.

This library can be used to center and normalize data, split
out testing data, train neural nets, and gauge performance of
trained models.

`QUICK SIGNATURES`

  |center|       mean-center `xss`; returns `(tensor, tensor)`
    ($xss$,        -tensor to center w/r to its 1st dimension
     $new_centers$ = `None`)
                 -first returned tensor has column means `new_`
                  `centers`; default is `new_centers` being zeros.

  |normalize|    normalize `xss`; returns `(tensor, tensor)`
    ($xss$,        -tensor to normalize w/r to its 1st dimension
     $new_widths$ = `None`,
                 -first tensor returned will now have columns
                  with st. devs `new_widths`; default is `new_`
                  `widths` being all ones.
     $unbiased$ = `True`)
                 -use n-1 instead of n in the denominator when
                  computing the standard deviation.

  |coh_split|    randomize and coherently split each tensor in
                 `*args`; returns `Tuple[tensor]`
    ($prop$,       -split like `prop`, 1 - `prop`
     $*args$,      -each of these tensors are split into two
     $randomize$ = `True`)
                 -whether to randomize before splitting.

  |train|        return `model` trained using SGD;
    ($model$,      -the instance of `nn.Module` to be trained
     $crit$,       -the criterion for assessing the loss
     $train_data$,
                 -either a tuple `(train_feats, train_targs)` or
                  `(train_feats, train_lengths, train_targs)`;
                  passing `train_lengths` or, below, `test_lengths`
                  is likely only relevant for recurrent nets.
     $test_data$ = `None`,
                 -either `(test_feats, test_targs)` or
                  `(test_feats, test_lengths, train_targs)`
     $learn_params$ = `{'lr': 0.1}`,
                 -a `dict` of the form `{'lr': 0.1,'mo': 0.9}` or
                  `{'lr': 0.1}`, or an instance of `LearnParams_`,
                  or an instance of `torch.optim.Optimizer`.
     $bs$ = `-1`,    -the mini-batch size; -1 is (full) batch
     $epochs$ = `10`,-train for this many epochs
     $graph$ = `0`,  -put 1 (or more) to show graph when training
     $print_lines$ = `(7,8)`,
                 -print 7 beginning lines and 8 ending lines;
                  put -1 to disable compressed printing.
     $verb$ = `2`,   -verbosity; 3 for more, 1 for less, 0 silent
     $gpu$ = `-1`,   -the gpu to run on, if any are available; if
                  none available, use the cpu; put -1 to use
                  the last gpu if multiple ones found; put -2
                  to override found gpu(s) and use the cpu.
                  Consider just accepting the default here.
     $valid_crit$ = `True`)
                 -function determining how the model is valida-
                  ted w/r to test data. The default results in
                  using `r_squared` for regression and `confusion_`
                  `matrix` for classification.

  |cross_validate_train|
    ($model$, $crit$, $train_data$, $k$, $**kwargs$)
     This is a helper function for `cross_validate`; each epoch
     it iterates fold-wise, validating on the `k` possible test
     sets, and returns the partially trained (for 1 epoch, by
     default) model; consider using `cross_validate` instead of
     calling this directly; the arguments are the same as those
     for `cross_validate`, but without `bail_after`.

  |cross_validate| return `model` cross-validate trained tupled
                 with the mean (`float`) of `model`'s validations
    ($model$,     -the model to be cross-validated
     $crit$,      -the loss function while training
     $train_data$,-either `(train_feats, train_targs)` or
                 `(test_feats, test_lengths, train_targs)`
     $k$ = `10`,    -the number of folds when cross-validating
     $bail_after$ = `5`,
                -bail after this many steps if no improvement
     $valid_crit$ = `None`,
                -the criterion to use when validating on test
                 data during cross validate training and on any
                 final testing data. Default `None` leads to us-
                 ing the loss function defined by `crit`.
     $cent_norm_feats$ = `(True, True)`,
                -whether to center and/or normalize features
     $cent_norm_targs$ = `(True, True)`,
                -whether to center and/or normalize targets
     $learn_params$ = `{'lr':0.1}`,
                -a `dict` of the form `{'lr': 0.1,'mo': 0.9}` or
                 `{'lr': 0.1}`, or an instance of `LearnParams_`, or
                 and instance of `torch.optim.Optimizer`.
     $bs$ = `-1`,   -the mini-batch size; -1 is (full) batch
     $epochs$ = `1`,-train for this many epochs during each run of
                 `cross_valid_train`.
     $verb$ = `1`,  -verbosity; 0 for silent
     $gpu$ = `-1`)  -the gpu to run on, if any are available; if
                 none available, use the cpu; put -1 to use the
                 last gpu if multiple ones found; put -2 to ov-
                 erride found gpu(s) and use the cpu.  Consider
                 just accepting the default here.

  |confusion_matrix| compute the confusion matrix when classify-
                 ing, say, `m` classes; returns `float`.
    ($prob_dists$,-these are the predictions of the model in the
                 form of a tensor (of shape `(n,m)`) of discrete
                 probability dists; this is normally just `mod-`
                 `el(test_feats)` or `model(train_feats)`
     $yss$        -a tensor of shape `(n)` holding the correct
                 classes
     $classes$,   -a tensor of shape `(n)` holding the possible
                 classes; normally this is `torch.arange(10)`, if
                 there are say 10 things being classified
     $return_error$ = `False`,
                -return error instead of proportion correct
     $show$ = `False`,
                -display the confusion matrix       -
     $gpu$ = `-1`,  -run on the fastest device, by default
     $class2name$ = `None`)
                -a dict mapping `int`s representing the classes
                 to the corresponing descriptive name (`str`)

  |r_squared|     return (`float`) the coefficient of determination
    ($yhatss$,    -either a trained model's best guesses (so of-
                 ten just `model(xss)`); or, a tuple of the form
                 `(model, xss)`. (Use the second form to execute
                 the model evaluation on the fastest device av-
                 ailable.)
     $yss$,       -the actual targets
     $gpu$ = `-1`,  -run on the fastest device, by default
     $return_error$ = `False`)

  |optimize_ols|  find optimal training hyper-parameters; returns
                a dict with keys 'lr' and 'mo'
    ($feats$,     -the `xss` for the data set
     $with_mo$ = `True`
                -if `False` just returns optimal learning rate
     $verb$ = `0`)  -default is silence; put 1 to include warnings,
                 and 2 to actually print out X^T*X where X is
                 the design matrix

  |copy_parameters| helper for sub-classing `LearnParams_`
    ($model$)     -copy the parameters of `model`

  |LearnParams_|  base class for defining learning parameters
    ($lr$ = `0.1`)  -we need at least a learning rate

  |Momentum|      subclass of `LearnParams_`, an instance of which
                adds momentum to gradient descent
    ($model$,     -model instance to which to add momentum
     $lr$ = `0.01`, -the desired learning rate
     $mo$ = `0.9`)  -the desired momentum

                    _____________________
"""
#Todo:
#  - consider removing datadevice from train.  Don't need it?
#  - look closely at 'center' for center and similar for normalize
#  - consider allowing train to just accept args. DONE?
#  - rewrite _Batcher to pull minibatches from dataset, and see if
#    that is as fast as pulling them from a tuple.
#  - consider adding functionality to train where gpu can be a neg
#    int, and saves image equivalent to that positive int instead
#    of displaying it.
#  - in the ducktyping, the Tensor types could be cleaned up, and
#    in docstrings it just says e.g. LongTensor
#  - Add to docs of confusion_matrix about passing in model too
#    for speed and about then no need to set classes.
#  - try to speed up evaluation by using model.eval() but be care
#    ful the dropout etc.
#  - Add notes to docstring about only fooling with testdate if
#    graphing, and doing so for speed.  <--IMPORTANT
#  - Fix the packing issue for minibatch in rec nets - graphing
#    against test loss on rec nets doesn't naturally work until
#    then (without accumulating it with a loop).
#  - Attempt to move to device only in train() and coh_split().
#    So try to refrain to going to device in programs (but still
#    get and pass the device, of course). THINK THROUGH THIS.
#    - what about the new normal -- only moving to device for in
#      the train fn -- w/r to cross_validate_train?
#    - grep this whole file for device and look carefully
#    - NOTE:  now using the gpu arg in train ...
#  - make cross_validate_train and cross_validate work with
#    variable length data
#    - add feats lengths to all three train fns and document-
#      ation
#  - Add option to train to show progress on training / testing
#    data each epoch.  Done for losses, but add another pane
#    to the graph with percentage correct training/testing.
#  - Implement stratified sampling at least when splitting out
#    testing data.  Maybe pass the relevant proportions to
#    coh_split.
#  - Try to catch last saved model or just continue on control-c
#    for, well, everything.
#    - Fix catch_sigint_and_break or remove it. If it is working
#      in bash, check and see how it interacts with interrupt in
#      say IDLE.
#  - Clean up verbosity in cross_validate_train and
#    cross_validate.
#  - Start type checking kwargs whenever everyone is clearly
#    running Python 3.6 or greater.
#  - Clean up strings by using multiline ones correctly. Use
#    verbosity so nothing is printed by default. Use
#    textwrap.dedent.
#  -  Use _check_kwargs everywhere in the other modules.
#     - this file is good
#  - fix asserts, like you did in r-squared, throughout
#  - catch tkinter error when bailing early on any graphing.
#  - optimize_ols is still somehow spitting out complex numbers
#    eigenvalues for terribly unconditioned but symmetric mats.
#  - write a cent_norm function and an un_norm_cent_wieghts that
#    reverts the weights of a trained linear model back to that of
#    the un_normalized un_centered data. Then use this for better
#    high degree poly regression.
#  - r-squared only really applies to ols regression (with linear
#    hypothesis) (*but what about poly linear regression*). Still
#    though, for a regression problem,  just split out testing
#    data and compute r-squared for that.
#  - when using cross-validation, just put in the means and st.devs.
#    of all the data when serializing (change doc strings in models)
#  - Build a graphing class
#  - Finish colorizing the confusion matrix
#  - use '\r' where possible in train printing (doesn't matter??)
#  - add option to confusion matrix to push through on gpu, like
#    you did for r_squared
# Done or didn't do:
#  - Add percentage or loss to ascii output in the presence of
#    testing data. DON"T DO THIS so that training without graph
#    will be fast.

import time
import functools
import tkinter
import copy
import torch
import torch.nn as nn
import torch.utils.data
from types import FunctionType
from typing import Dict
from textwrap import dedent
import du.utils

__author__ = 'Scott Simmons'
__version__ = '0.9'
__status__ = 'Development'
__date__ = '01/23/20'
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

# glom together some types
IntTensor = (torch.ShortTensor, torch.IntTensor, torch.LongTensor,
    torch.cuda.ShortTensor, torch.cuda.IntTensor, torch.cuda.LongTensor)
FloatTensor = (torch.HalfTensor, torch.FloatTensor, torch.DoubleTensor,
    torch.cuda.HalfTensor, torch.cuda.FloatTensor, torch.cuda.DoubleTensor)

def center(xss, new_centers = None):
  """Center a tensor.

  With this you can translate (with respect to the first dimen-
  sion) data to anywhere. If the `new_centers` is `None`, then
  this simply mean-centers the data along the first dimension;
  in other words, it rigidly translates `xss` so that its mean
  along the first dimension is the zero tensor.

  Notice that the returned object is a tuple. So if you want to
  simply mean-center a tensor you would call this function like
               `xss_centered, _ = center(xss)`
  That is, you can use an underscore (or whatever) if you don't
  need the means.

  Args:
    $xss$ (`torch.Tensor`) The tensor to center.
    $new_centers$ (`torch.Tensor`) A tensor, the number of dimen-
        sions of which is one less than that of `xss` and whose
        shape is in fact `(d_1,`...`,d_n)` where `xss` has as its
        shape `(d_0, d_1,`...`,d_n)`. The default is `None` which is
        equivalent to `new_center` being the zero tensor.

  Returns:
    `(torch.Tensor, torch.Tensor)`. A tuple of tensors the first
        of which is `xss` centered with respect to the first dim-
        ension; the second is a tensor the size of the remain-
        ing dimensions and that holds the means.

  >>> `xss = torch.arange(12.).view(3,4); xss`
  tensor([[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.]])
  >>> `xss, xss_means = center(xss)`
  >>> `xss, xss_means`
  (tensor([[-4., -4., -4., -4.],
          [ 0.,  0.,  0.,  0.],
          [ 4.,  4.,  4.,  4.]]), tensor([4., 5., 6., 7.]))
  >>> `xss_, _ = center(xss, -xss_means)`
  >>> `int(torch.all(torch.eq(xss, xss_)).item())`
  1

  >>> `xss = torch.arange(12.).view(3,2,2)`
  >>> `xss, xss_means = center(xss)`
  >>> `xss_means.shape`
  torch.Size([2, 2])
  >>> `xss_, _ = center(xss, -xss_means)`
  >>> `int(torch.all(torch.eq(xss, xss_)).item())`
  1
  """
  xss_means = xss.mean(0)
  if isinstance(new_centers, torch.Tensor):
    assert new_centers.size() == xss_means.size(),\
        'new_centers must have size {}, not {}'.\
            format(xss_means.size(),new_centers.size())
  else:
    new_centers = xss_means
  return xss - new_centers, xss_means

def normalize(xss, new_widths = None, unbiased = True):
  """Normalize without dividing by zero.

  See the documentation for the function `center`. This is com-
  pletely analagous.

  Args:
    $xss$ (`torch.Tensor`)
    $new_widths$ (`torch.Tensor`)
    $unbiased$ (`bool`): If unbiased is `False`, divide by `n` instead
        of `n-1` when computing the standard deviation. Default:
        True.

  Returns:
    `(torch.Tensor, torch.Tensor)`. A tuple of tensors the first
        of which is `xss` normalized with respect to the first
        dimension, except that those columns with standard dev
        less than a threshold are left unchanged. The list of
        standard devs, with numbers less than the threshold re-
        placed by 1.0, is the second tensor returned.

  >>> `xss = torch.tensor([[1, 2, 3], [6, 7, 8]]).float()`
  >>> `xss, _ = normalize(xss, unbiased = False)`
  >>> `xss.tolist()`
  [[0.4...
  >>> `xss = torch.tensor([[1, 2, 3], [1, 7, 3]]).float()`
  >>> `xss, _ = normalize(xss, unbiased = False)`
  >>> `xss.tolist()`
  [[1.0...
  """
  # add and assert checking that new_width is right dim.
  xss_stdevs = xss.std(0, unbiased)
  xss_stdevs[xss_stdevs < 1e-7] = 1.0
  if isinstance(new_widths, torch.Tensor):
    new_xss = xss.div_(new_widths)
  else:
    new_xss = xss.div_(xss_stdevs)
  return new_xss, xss_stdevs

def coh_split(prop, *args, **kwargs):
  """Coherently randomize and split tensors.

  This splits each tensor in `*args` with respect to the first
  dimension. First, the tensors are randomized with the respect
  to their first dimension. The same random permutation is app-
  lied to each tensor (hence the word 'coherent' in this func-
  tions name).

  Args:
    $prop$ (`float`): The proportion to split out. Suppose this is
        0.8. Then for each pair in the return tuple, the first
        tensor holds 80% of the data and the second holds the
        other 20%.
    $*args$ (`torch.tensor`): The tensors to be randomized and
        split; each must have the same length in the first dim-
        ension.

  Kwargs:
    $randomize$ (`bool`): Whether to randomize before splitting.
        Default: `True`

  Returns:
    `Tuple[torch.tensor]`. A tuple of length twice that of `args`
        and holding, in turn, pairs, each of which is a tensor
        in `args` split according to `prop`.

  >>> `from torch import rand`
  >>> `coh_split(0.6, rand(2,3), rand(3,3))`
  Traceback (most recent call last):
    ...
  AssertionError: all tensors must have same size in first dim
  >>> `xss=rand(4, 2); xss_lengths=rand(4); yss=rand(4, 3)`
  >>> `len(coh_split(0.6, xss, xss_lengths, yss))`
  6
  >>> `xss_train, xss_test, *_ = coh_split(0.75, xss, yss)`
  >>> `xss_train.size()`
  torch.Size([3, 2])
  """
  du.utils._check_kwargs(kwargs,['randomize'])
  randomize = kwargs.get('randomize',True)
  assert 0 <= prop <= 1, dedent("""\
      Arg prop ({}) must be between 0 and 1, inclusive.
  """.format(prop))
  len_ = list(map(len, args))
  assert all(len_[0] == x for x in len_), "all tensors must have same size "+\
      "in first dim"
  if randomize:
    indices = torch.randperm(len_[0])
    args = [tensor.index_select(0, indices) for tensor in args]
  cutoff = int(prop * len_[0])
  split_args = [[tensor[:cutoff], tensor[cutoff:]] for tensor in args]
  return_args =[item for sublist in split_args for item in sublist]
  return tuple(return_args)

def copy_parameters(model):
  """Copy a models parameters.

  This is a helper function to copy a model's parameters and
  initialize each copied tensor so as to hold all zeros. The
  returned tensors reside on the same device as that of the
  corresponding tensor in model.parameters().

   Args:
     $model$ (`nn.Module`): The model whose parameters to copy.

   Returns:
     `List[tensor]`: A list with the structure that matches exac-
         tly that of `model.parameters()` (except that it's a list
         instead of a generator) but with its tensors initiali-
         zed to be all zeros.
  """
  params = []
  for param in model.parameters():
    params.append(param.data.clone())
  for param in params: param.zero_()
  return params

class LearnParams_:
  """The base class for adaptive learning schemes.

  This implements the minimal gradient update scheme during
  SGD; namely, multiplying the gradient by a smallish `learning`
  `rate`.

  Args:
    $lr$ (`float`): The learning rate.
  """
  def __init__(self, lr = 0.1):
    """Constructor.

    Set the instance variable `self.lr`.
    """
    self.lr = lr

  def __str__(self):
    """Make a string representation."""
    return 'learning rate: ' + du.utils.format_num(self.lr)

  def set_device(self, device):
    """`pass` here, but sub-classes might want this."""
    pass

  def update(self, parameters):
    """Update parameters.

    This implements the standard update rule for gradient
    descent: i.e.,

      `for param in parameters:`
        `param.data.sub_(self.lr * param.grad.data)`

    Args:
      $parameters$ (`generator`): The model parameters (in the
          form of an iterator over tensors) to be updated.
    """
    for param in parameters:
      param.data.sub_(self.lr * param.grad.data)

class Momentum(LearnParams_):
  """Add momentum to gradient descent.

  If an instance of this is passed to `du.lib.train` then, during
  training, the update rule in SGD incorporates momentum.
  """
  def __init__(self, model, lr = 0.01, mo = 0.9):
    """Constructor.

    Set instance variables `lr` and `mo` and create an instance
    variable `z_params` which is essentially a zeroed out (in-
    itially) clone of `model.parameters()`.

    Args:
      $lr$ (`float`): The learning rate during training.
      $mo$ (`float`): The momentum during training.
    """
    super().__init__(lr)
    self.mo = mo
    self.z_params = copy_parameters(model)

  def __str__(self):
    """Append momentum info to string rep of the base class."""
    return super().__str__() + ', momentum: ' + du.utils.format_num(self.mo)

  def set_device(self, device):
    """Send `z_params` to live on device."""
    for param in self.z_params:
      param = param.to(device)

  def update(self, params):
    """Update the learning hyper-parameters.

    The learning hyper-parameters now include momentum so
    the update rule here is accordingly enhanced.

    Args:
      $parameters$ (`generator`): The parameters (in the form of
          an iterator of tensors) to be updated.
    """
    for i, (z_param, param) in enumerate(zip(self.z_params, params)):
      self.z_params[i] = z_param.mul_(self.mo).add_(param.grad.data)
      param.data.sub_(self.z_params[i] * self.lr)

def _parse_data(data_tuple, device = 'cpu'):
  """Simple helper function for the train function.

  Args:
    $data_tuple$ (`Tuple[tensor]`): Length either 2 or 3.

  Returns:
    `Tuple[tensor]`.
  """
  feats = data_tuple[0].to(device); targs = data_tuple[-1].to(device)
  if len(data_tuple) == 3:
    feats_lengths = data_tuple[1].to(device)
    assert len(feats_lengths) == len(feats),\
        "No. of feats lengths ({}) must equal no. of feats ({}).".\
            format(len(feats_lengths), len(feats))
  else:
    assert len(data_tuple) == 2, 'data_tuple must have len 2 or 3'
    feats_lengths = None
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  return feats, feats_lengths, targs

#def _batcher(data_tuple, bs, data_device, model_device):
#  """Helper function for the train function that returns a gen-
#  erator which, after the data are coherently randomized, kicks
#  out batches of the specified size.
#
#  Args:
#    $data_tuple$ (`Tuple[tensor]`): The tensors to be coherent-
#        ly batched.
#    $bs$ (`int`): The batchsize.
#    $data_device$ (`Union[str, torch.device]`): The device on which
#        to batch the tensors from.
#    $model_device$ (`Union[str, torch.device]`): The device to move
#        the batches to just before yielding them.
#
#  Returns:
#    `generator`. A generator that yields batches in the form of
#        tuples of the same length as `data_tuple`.
#  """
#  num_examples = len(data_tuple[0])
#  tuple([t.to(data_device) for t in data_tuple])
#  indices = torch.randperm(num_examples, device = data_device)
#  for idx in range(0, num_examples, bs):
#    yield tuple([t.index_select(0,indices[idx: idx + bs]).to(model_device)\
#        for t in data_tuple])

class _MiniBatcher:
  """Helper class for the train function.

  An instance of this can be used in the same way that one uses
  an instance of `DataLoader`.
  """
  class _Dataset(torch.utils.data.Dataset):
    def __init__(self, tup):
      self.tuple = tup
      self.features = tup[0]
      self.targets = tup[-1]
    def __len__(self):
      return len(self.tuple[0])
    def __getitem__(self, idx):
      return tuple(t[idx] for t in self.tuple)

  #def __init__(self, data_tuple, bs, data_device, model_device):
  def __init__(self, data_tuple, bs, data_device):
    """
    Args:
      $data_tuple$ (`Tuple[tensor]`): The tensors to be coher-
          ently batched.
      $bs$ (`int`): The batchsize.
      $data_device$ (`Union[str, torch.device]`): The device on
          which to batch the tensors from.
      $model_device$ (`Union[str, torch.device]`): The device
          to move the minibatches to just before returning
          them.
    """
    self.tuple = tuple([t.to(data_device) for t in data_tuple])
    self.dataset = self._Dataset(self.tuple)
    self.bs = bs
    self.data_device = data_device
    #self.model_device = model_device
    self.indices = torch.randperm(len(self.dataset), device = data_device)

  def __iter__(self):
    self.idx = 0
    return self

  def __next__(self):
    if self.idx >= len(self.dataset) - 1:
      self.idx = 0
      self.indices = torch.randperm(len(self.dataset), device=self.data_device)
      raise StopIteration
    #minibatch = tuple([t.index_select(0,
    #    self.indices[self.idx: self.idx + self.bs]).to(self.model_device)\
    #        for t in self.tuple])
    minibatch = tuple([t.index_select(0,
        self.indices[self.idx: self.idx + self.bs]) for t in self.tuple])
    self.idx += self.bs
    return minibatch

def train(model, crit, train_data, **kwargs):
  """Train a model.

  The loss printed to the console is the average loss per samp-
  le over an epoch as that average is accumulated during train-
  ing. If the number of training examples is divisible by the
  batchsize then, during training, the model sees each example
  in the training data exactly once.

  !Notes on specifying training hyper-parameters!

  The argument `learn_params` specifies the training hyper-para-
  meters. It can be constructed in one of three ways. To train
  with constant learning rate and momentum, one passes a simple
  dictionary; either, for example,

       train( ..., `learn_params = {'lr': 0.01}`, ...)

  or, e.g.,

    train( ..., `learn_params = {'lr': 0.01, 'mo': 0.9}`, ...).

  Alternatively, `learn_params` can be an instance of (a subclass
  of) the `LearnParams_` class or an instance of `torch.optim.Opti`
  `mizer`. (Type `pd du.examples` and scroll down to the !simple lin!
  !ear regression with learning rate decay! section to see an ex-
  ple that uses the `LearnParams_` class.)

  !Notes on training with a GPU!

  In the presence of at least one GPU, the `gpu` argument can be
  used to move some or all computations to the GPU(s). Generic-
  ally one can accept the default (`gpu` = `(-1,)`) which sends all
  computations to the (last of any) found GPU(s) and, if there
  are no GPU(s), to the (first) CPU (thread).

  Just before mini-batches are forward-passed through the model
  during training, they are moved from the CPU to the training
  device determined by the first entry in the tuple `gpu`. Mean-
  while, the model has always been moved to the training device
  at the beginning of training.

  !Note on validation and efficiency!

  In order to provide an option that trains as efficiently as
  possible, unless `graph` is positive, any test data is ignored;
  that is, the model is simply trained on the provided training
  data, and the loss per epoch is displayed to the console. Use
  the default `gpu = (-1,)` to train on the fastest available de-
  vice.

  You can set `graph` to be positive (and forego testing data) in
  order to real-time graph the losses per epoch at cost in time
  but at no cost in VRAM (assuming you have GPU(s)) if you set
  `gpu = (-1, -2)`. Here the -1 leads to training on the GPU and
  the -2 causes validation during training to take place on the
  CPU. Moreover, the training data is immediately copied to the
  CPU, thus freeing VRAM for training (at the expense of time
  efficiency since the validation is slower on a CPU).

  By default, any provided `test_data` resides on the device on
  which training occurs. In a bind (such as running out of VRAM
  when training on a GPU) one can again set `gpu = (-1,-2)` which
  causes `model` to, after each training loop, be deep copied to
  the CPU and evaluated on test_data (which resides in CPU mem-
  ory). Of course, there is added cost and hence slow down when
  deep copying and evaluating on the CPU.

  On a machine with more than one GPU, one can also try setting
  `gpu = (0, 1)` with potentailly less slowdown.

                    _____________________

  Args:
    $model$ (`nn.Module`): The instance of Module to be trained.
    $crit$ (`nn.modules.loss`): The loss function when training.
    $train_data$ (`Tuple[torch.Tensor]`): A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tensor is on-
        ly necessary when training a recurrent net on variable
        length inputs. In that case, the triple of tensors must
        be of the form
           `(train_features, train_lengths, train_targets)`;
        i.e., the first tensor holds the inputs of the training
        data, the second holds the corresponding lengths, and
        the third holds the training data outputs.

        If the data are not of variable length, then there is
        no need to pass the middle tensor in the triple above;
        so one passes just
                  `(train_features, train_targets)`.
        In any case, `train_features` must have dimension greater
        than or equal to 2, while `train_targets` should be 2-dim
        in the case of a regression problem (i.e., if it holds
        floating point numbers) and 1-dim for a classification
        problem (in which case it hold integers). `train_lengths`
        should always be 1-dim. (Type `pd du.examples` to read a-
        bout conventions and assumptions surrounding training/
        testing data.)

  Kwargs:
    $test_data$ (`Tuple[torch.Tensor]`): (Optional) data on which
        to test, in the form of a tuple of length 2 or 3; i.e.,
        that matches the length of `train_data`. The loss on test
        data is computed each epoch. However, The test data is
        not shown to the model as part of backpropagation. Def-
        ault: `None`.
    $learn_params$
        (`Union[dict,LearnParam_, torch.optim.Optimizer]`): The
        training, or 'learning', hyperparameters in the form of
        an instance of `LearnParams_`; or, for basic functionali-
        ty, a `dict` that maps the string 'lr', and optionally
        'mo', to `float`s; or an instance of `torch.optim.Optimizer`
        Default: `{'lr': 0.1}`.
    $bs$ (`int`): The mini-batch size where -1 forces batch gradi-
        ent descent (i.e. feed-forwarding all training examples
        before each back-propagation). Default: -1.
    $epochs$ (`int`): The number of epochs to train over, where an
        epoch is the 'time' required to see each training exam-
        ple exactly once. Default: `10`.
    $graph$ (`int`): If positive then, during training, display a
        real-time graph. If greater than 1, then the beginning
        `graph` number of losses are thrown away when displaying
        the graph at the completion of training. Displaying a
        graph at all requires `matplotlib`, and a running X serv-
        er). If 0, do not display a graph. Default: `0`.
    $print_lines$ (`Tuple[int]`): A tuple, the first component of
        which is the number of losses to print before/after the
        ellipses during compressed printing to the console. A
        length one tuple is duplicated into a length two one.
        Put (-1,) to print all losses. Default: `(7,)`.
    $verb$ (`int`): Verbosity; 0 = silent, ... , 3 = all. Def.: `2`.
    $gpu$ (`Tuple[int]`): Tuple of `int`s of length 1 or 2 where the
        first entry determines the device to which the model is
        moved and, in fact, on which the forwarding and back-
        propagation through the model takes place. The second
        entry determines the device to which the model is deep
        copied (if necessary) for the purpose of validation ag-
        aisnt any test data present. If this is a length 1 tup-
        le, then that number is used to determine both devices.

        If no GPUs are present, then accept the default. Other-
        wise an `int` determines the GPU to use for training/val-
        idating. When GPU(s) are present, set an entry of the
        tuple to an `int` to select the corresponding GPU, or
        to -1 to use the last GPU found (and to use the CPU if
        no GPU is found), or to -2 to override using a found
        GPU and instead use the CPU. Default: `(-1, -1)`.
    $valid_crit$ (`Union(bool, function)`): If `graph` is positive,
        and `test_data` is present then, while graphing, `valid_`
        `crit` is applied to test data after every epoch, and the
        output is displayed on the graph. The default (which is
        `True`) results in automatically using `r-squared` if the
        target data are floating point tensors, and to using `co`
        `nfusion_matrix` if the targets are tensors of integers.

        Any function that maps the outputs of `model` (e.g., `yhat`
        `ss` or `prob_dists`) along with `yss` to a float can be
        provided for `valid_crit`.
    $args$ (`argparse.Namespace`): With the exception of `test_data`
        `valid_crit`, and this argument, all `kwargs` can be passed
        in via attributes (of the same name) of an instance of
        `argparse.Namespace`. Default: None.

  Returns:
    `nn.Module`. The trained model (still on the device determin-
        ed by `gpu`).
  """
  # this is train
  # check and process kwargs
  du.utils._check_kwargs(kwargs,['test_data','learn_params','bs','epochs',
      'graph','print_lines','verb','gpu','valid_crit','args'])
  test_data = kwargs.get('test_data', None)
  args = kwargs.get('args', None)
  if 'args' == None:
    class args: pass # a little finesse if args wasn't passed

  learn_params = kwargs.get('learn_params',
      {'lr': 0.1 if not hasattr(args,'lr') else args.lr,
          'mo': 0.0 if not hasattr(args,'mo') else args.mo} if \
          not hasattr(args,'learn_params') else args.learn_params)
  bs = kwargs.get('bs', -1 if not hasattr(args,'bs') else args.bs)
  epochs=kwargs.get('epochs', 10 if not hasattr(args,'epochs') else args.epochs)
  print_lines = kwargs.get('print_lines',
      (7,) if not hasattr(args,'print_lines') else args.print_lines)
  if len(print_lines) > 1:
    print_init, print_last = print_lines
  else:
    print_init, print_last = print_lines[0], print_lines[0]
  verb = kwargs.get('verb', 2 if not hasattr(args,'verb') else args.verb)
  graph = kwargs.get('graph', 0 if not hasattr(args,'graph') else args.graph)
  gpu = kwargs.get('gpu', (-1,) if not hasattr(args,'gpu') else args.gpu)
  valid_crit = kwargs.get('valid_crit', True)

  start = time.time() # start (naive) timing here

  graph = 1 if graph == True else graph
  assert graph>=0, 'graph must be a non-negative integer, not {}.'.format(graph)

  # get devices determined by the arg gpu
  if isinstance(gpu, (tuple,list)) and len(gpu) == 1: gpu = (gpu[0], gpu[0])
  else: assert isinstance(gpu, (tuple,list)) and len(gpu) > 1
  model_device = du.utils.get_device(gpu[0])  # where the training takes place
  valid_device = du.utils.get_device(gpu[1])  # where validation happens
  data_device = torch.device('cpu',0)
  if verb > 0:
    print('training on {} (data on {})'.format(model_device, data_device),end='')
    if valid_crit and graph>0: print('; validating on {}'.format(valid_device))
    else: print()

  # parse the training data and leave it in data_device memory
  if isinstance(train_data, torch.utils.data.DataLoader):
    has_lengths = True if len(train_data.dataset[0]) > 2 else False
    num_examples = len(train_data.dataset)
  else:
    assert 2 <= len(train_data) <= 3
    assert all([isinstance(x, torch.Tensor) for x in train_data])
    has_lengths = True if len(train_data) > 2  else False
    num_examples = len(train_data[0])
    train_data = _MiniBatcher(train_data, bs, data_device)
  assert hasattr(train_data.dataset, 'features') and \
      hasattr(train_data.dataset, 'targets')


  if bs <= 0: bs = num_examples

  model = model.to(model_device) # move the model to the right device
  if verb > 2: print(model)

  # process learn_params
  has_optim = False
  if isinstance(learn_params, Dict):
    for key in learn_params.keys(): assert key in ['lr','mo'],\
        "keys of learn_params dict should be 'lr' or 'mo', not {}.".format(key)
    assert 'lr' in learn_params.keys(), "input dict must map 'lr' to  a float"
    lr = learn_params['lr']
    if verb > 1: print('learning rate:', du.utils.format_num(lr), end=', ')
    if 'mo' not in learn_params.keys():
      learn_params = LearnParams_(lr = lr)
      mo = None
    else:
      mo = learn_params['mo']
      if verb > 1: print('momentum:', du.utils.format_num(mo), end=', ')
      learn_params = Momentum(model, lr = lr, mo = mo)
      learn_params.set_device(model_device)
    if verb > 1: print('batchsize:', bs)
  elif isinstance(learn_params, torch.optim.Optimizer):
    has_optim = True
  else:
    assert isinstance(learn_params, LearnParams_), dedent("""\
        learn_params must be a dict or an instance of a subclass of
        LearnParams_, not a {}.""".format(type(learn_params)))
    # set the device for learn params
    learn_params.set_device(model_device)
    if verb > 1: print(learn_params, end=', ')
    if verb > 1: print('batchsize:', bs)

  # setup valid_crit
  if isinstance(valid_crit, bool):
    if valid_crit:
      if isinstance(train_data.dataset[0][-1], FloatTensor):
        valid_crit = lambda yhatss, yss: r_squared(yhatss,yss,gpu=valid_device)
      elif isinstance(train_data.dataset[0][-1], IntTensor):
        valid_crit = lambda prob_dists, yss:\
            confusion_matrix(prob_dists, yss, gpu=valid_device)
      else:
        raise RuntimeError('please specify a function to use for validation')
  else:
    assert isinstance(valid_crit, FunctionType)

  if graph:
    # don't import this until now in case someone no haz matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    plt.ion()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch', size='larger')
    ax1.set_ylabel('average loss',size='larger')
    ax2 = ax1.twinx()
    ax2.set_ylabel('validation',size='larger');
    xlim_start = 1

    # once and for all clone and move train data for validation purposes if nec.
    # is will break if not Dataloader
    if valid_device == data_device:
      train_feats_copy = train_data.dataset.features
      train_targs_copy = train_data.dataset.targets
    else:
      train_feats_copy, train_targs_copy = \
          (train_data.dataset.features.detach().clone().to(valid_device),\
              train_data.dataset.targets.detach().clone().to(valid_device))
    #train_feats_copy, train_targs_copy = (train_feats, train_targs) if\
    #    valid_device == data_device else\
    #        (train_feats.detach().clone().to(valid_device),\
    #            train_targs.detach().clone().to(valid_device))

    # these will hold the losses and validations for train data
    losses = []
    if valid_crit: v_dations = []

    # only bother with the following if there are test_data and valid_crit
    if test_data and valid_crit:
      if len(test_data[0]) == 0: test_data = None
      else:
        # once and for all put the test data on valid_device
        test_feats, test_feats_lengths, test_targs =\
            _parse_data(test_data, valid_device)
        num_test_examples = len(test_data)

        # this will hold the losses for test data
        losses_test=[]

      # this will hold the validations for test data
      v_dations_test = []

  # set up console printing
  if print_init == -1 or print_last == -1: print_init, print_last = epochs, -1

  # try to catch crtl-C
  du.utils._catch_sigint()

  # training loop
  for epoch in range(epochs):
    accum_loss = 0
    #this breaks if not Dataloader
    #for batch in _batcher(train_data, bs, data_device, model_device):
    for batch in train_data:
      loss = crit(model(
          *map(lambda x: x.to(model_device), batch[:-1])),
          batch[-1].to(model_device))
      accum_loss += loss.item()
      if has_optim: learn_params.zero_grad()
      else: model.zero_grad()
      loss.backward()

      if has_optim: learn_params.step()
      else: learn_params.update(model.parameters())

    # print to terminal
    if print_init * print_last != 0 and verb > 0:
      loss_len = 20
      base_str = "epoch {0}/{1}; loss ".format(epoch+1, epochs)
      loss_str = "{0:<10g}".format(accum_loss*bs/num_examples)
      if epochs < print_init+print_last+2 or epoch < print_init:
        print(base_str + loss_str)
      elif epoch > epochs - print_last - 1:
        print(end='\b'*len(base_str))
        print(base_str + loss_str)
      elif epoch == print_init:
        print("...")
      else:
        print(' '*loss_len, end='\b'*loss_len)
        print(end='\b'*len(base_str))
        loss_len = len(loss_str)
        print(base_str+loss_str, end='\b'*loss_len, flush=True)

    if graph:
      losses.append(accum_loss*bs/num_examples)
      # copy the model to the valid_device, if necessary
      model_copy = model if valid_device == model_device else\
          copy.deepcopy(model).to(valid_device)
      v_dations.append(valid_crit(model_copy(train_feats_copy),train_targs_copy))
      if test_data is not None:
        if has_lengths:
          loss=crit(model_copy(test_feats,test_feats_lengths),test_targs).item()
        else:
          loss = crit(model_copy(test_feats), test_targs).item()
        losses_test.append(loss)
        v_dations_test.append(valid_crit(model_copy(test_feats), test_targs))
      if epoch > epochs - graph:
        xlim_start += 1
      ax1.clear()
      ax2.clear()
      ax1.set_xlabel('epoch', size='larger')
      ax1.set_ylabel('average loss',size='larger')
      ax2.set_ylabel('validation',size='larger')
      xlim = range(xlim_start,len(losses)+1)
      loss_ys = np.array(losses[xlim_start-1:])
      v_dation_ys = np.array(v_dations[xlim_start-1:])
      if test_data:
        losstest_ys = np.array(losses_test[xlim_start-1:])
        v_dationtest_ys = np.array(v_dations_test[xlim_start-1:])
        ax1.plot(xlim,losstest_ys,xlim,loss_ys,color='black',lw=.5)
        ax1.fill_between(xlim,losstest_ys,loss_ys,where = losstest_ys >=loss_ys,
            facecolor='tab:red',interpolate=True, alpha=.8)
        ax1.fill_between(xlim,losstest_ys,loss_ys,where = losstest_ys <=loss_ys,
            facecolor='tab:blue',interpolate=True, alpha=.8)
        ax2.plot(xlim,v_dationtest_ys,xlim,v_dation_ys,color='black',lw=.5)
        ax2.fill_between(xlim,v_dationtest_ys,v_dation_ys,
            where = v_dationtest_ys >=v_dation_ys, facecolor='tab:red',
            interpolate=True, alpha=.8,label='test > train')
        ax2.fill_between(xlim,v_dationtest_ys,v_dation_ys,
            where = v_dationtest_ys <=v_dation_ys,
            facecolor='tab:blue',interpolate=True, alpha=.8,label='train > test')
        ax2.legend(fancybox=True, loc=2, framealpha=0.8, prop={'size': 9})
      else:
        ax1.plot(xlim,loss_ys,color='black',lw=1.2,label='loss')
        ax2.plot(xlim,v_dations,color='tab:blue',lw=1.2,label='validation')
        ax1.legend(fancybox=True, loc=8, framealpha=0.8, prop={'size': 9})
        ax2.legend(fancybox=True, loc=9, framealpha=0.8, prop={'size': 9})
      len_test_data = len(test_data[0]) if test_data is not None else 0
      plt.title('training on {} ({:.1f}%) of {} examples'.format( num_examples,
          100*(num_examples/(num_examples+len_test_data)),
          num_examples+len_test_data))
      try:
        fig.canvas.flush_events()
      except tkinter.TclError:
        plt.ioff()
        exit()
      fig.tight_layout()

  end = time.time()
  if verb > 0: print ('trained in {:.2f} secs'.format(end-start))

  if graph:
    plt.ioff()
    plt.title('trained on {} ({:.1f}%) of {} examples'.format(num_examples,
        100*(num_examples/(num_examples+len_test_data)),
        num_examples+len_test_data))
    fig.tight_layout()
    plt.show(block = True)

  #model = model.to('cpu')
  return model

def cross_validate_train(model, crit, train_data, k, **kwargs):
  """Cross-validate train a model for one (by default) epoch.

  Rather than calling this directly, consider calling the func-
  tion `cross_validate` in this module.

  The data are appropriately centered and normalized during
  training. If the number of the features is not divisible by
  `k`, then the last chunk is thrown away (so make the length of
  it small, if not zero).

  Args:
    $model$ (`nn.Module`): The instance of `nn.Module` to be trained.
    $crit$ (`nn.modules.loss`): The loss function when training.
    $train_data$ (`Tuple[torch.Tensor]`): A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tuple is only
        necessary when training a recurrent net on variable
        length inputs. In that case, the triple of tensors must
        be of the form `(train_features, train_lengths, train_`
        `targets)`. That is, the first tensor holds the inputs of
        the training data, the second holds the corresponding
        lengths, and the third holds the training data outputs.
        If the data are not of variable length, then there is
        no need to pass the middle tensor in the triple above;
        so one passes `(train_features, train_targets)`. In any
        case, each of the tensors in the tuple must be of dim-
        ension at least 2, with the first dimension indexing
        the training examples.
    $k$ (`int`): The number of folds on which to cross-validate.
        Default: `10`.

  Kwargs:
    $valid_crit$ (`nn.Module`): The validation criterion to use
        when gauging the accuracy of the model on test data.
        If `None`, this is set to `crit`, the training criterion.
        Default: `None`.
    $cent_norm_feats$ (`Tuple[bool]`): Tuple with first entry det-
        ermining whether to center the features; and the sec-
        ond, whether to normalize them. Default: `(True, True)`.
    $cent_norm_targs$ (`Tuple[bool]`): Tuple with first entry det-
        ermining whether to center the targets, and the second,
        whether to normalize them. Default: `(True, True)`.
    $learn_params$
        (`Union[dict,LearnParams_,torch.optim.Optimizer]`):
        The training or 'learning' hyperparameters in the form
        of an instance of the class `LearnParams_`; or, for bas-
        ic functionality, a `dict` whose keys map the string
        'lr', and optionally 'mo', to `float`s; or an instance of
        `torch.optim.Optimizer`. Default: `{'lr': 0.1}`.
    $bs$ (`int`): The mini-batch size where -1 forces batch grad-
        ient descent (i.e. feed-forwarding all training exam-
        ples before each backpropagation). Default: `-1`.
    $epochs$ (`int`): The number of epochs to train over for each
        validation step. Default: `1`.
    $verb$ (`int`): The verbosity. 0: silent, 1: more, 2: all. De-
        fault: `2`.
    $gpu$ (`int`): Which gpu to use in the presence of one or more
        gpus, where -1 means to use the last gpu found, and -2
        means to override using a found gpu and use the cpu.
        Default: `-1`.

  Returns:
    `nn.Module`. Returns `model` which has been partially trained
        (for one epoch, by default) along with a tensor holding
        its `k` validations.
  """
  #_this is cross_validate_train
  du.utils._check_kwargs(kwargs,['k','valid_crit','cent_norm_feats',\
      'cent_norm_targs','learn_params','bs','epochs','gpu','verb'])
  du.utils._catch_sigint()
  valid_crit = kwargs.get('valid_crit', None)
  assert 2 <= len(train_data) <= 3, dedent("""\
      Argument train_data tuple must have length 2 or 3, not {}
  """.format(len(train_data)))
  feats_lengths = None
  if len(train_data) == 2:
    feats, targs = train_data
  elif len(train_data) == 3:
    feats, feats_lengths, targs = train_data
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  assert not feats_lengths, 'variable length not implemented yet'
  cent_feats, norm_feats = kwargs.get('cent_norm_feats',(True, True))
  cent_targs, norm_targs = kwargs.get('cent_norm_targs',(True, True))
  learn_params = kwargs.get('learn_params', {'lr': 0.1})
  bs = kwargs.get('bs', -1); epochs = kwargs.get('epochs', 1)
  verb = kwargs.get('verb', 2); gpu = kwargs.get('gpu', -1)

  valids = torch.zeros(k) # this will hold the k validations
  chunklength = len(feats) // k

  if not valid_crit: valid_crit = crit

  # randomize
  indices = torch.randperm(len(feats))
  xss = feats.index_select(0, indices)
  yss = targs.index_select(0, indices)

  for idx in range(0, chunklength * (len(xss)//chunklength), chunklength):

    xss_train = torch.cat((xss[:idx],xss[idx+chunklength:]),0).clone()
    xss_test = xss[idx:idx + chunklength].clone()
    yss_train = torch.cat((yss[:idx],yss[idx+chunklength:]),0).clone()
    yss_test = yss[idx:idx + chunklength].clone()

    if cent_feats: xss_train, xss_train_means = center(xss_train)
    if norm_feats: xss_train, xss_train_stdevs = normalize(xss_train)
    if cent_targs: yss_train, yss_train_means = center(yss_train)
    if norm_targs: yss_train, yss_train_stdevs = normalize(yss_train)

    model = train(
        model=model,
        crit=crit,
        train_data=(xss_train, yss_train),
        learn_params = learn_params,
        bs=bs,
        epochs=epochs,
        verb=verb-1,
        gpu=gpu)

    if cent_feats: xss_test.sub_(xss_train_means)
    if norm_feats: xss_test.div_(xss_train_stdevs)
    if cent_targs: yss_test.sub_(yss_train_means)
    if norm_targs: yss_test.div_(yss_train_stdevs)

    valids[idx//chunklength] = valid_crit(model(xss_test), yss_test)

  return model, valids

def cross_validate(model, crit, train_data, k, **kwargs):
  """Cross-validate a model.

  The data are appropriately centered and normalized during
  training. If the number of the features is not divisible by
  `k`, then the last chunk is thrown away (so make the length of
  it small, if not zero).

  Args:
    $model$ (`nn.Module`): The instance of `nn.Module` to be trained.
    $crit$ (`nn.modules.loss`): The loss function when training.
    $train_data$ (`Tuple[torch.Tensor]`): A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tuple is only
        necessary when training a recurrent net on variable
        length inputs. In that case, the triple of tensors must
        be of the form `(train_features, train_lengths, train_`
        `targets)`. That is, the first tensor holds the inputs of
        the training data, the second holds the corresponding
        lengths, and the third holds the training data outputs.
        If the data are not of variable length, then there is
        no need to pass the middle tensor in the triple above;
        so one passes `(train_features, train_targets)`. In any
        case, each of the tensors in the tuple must be of dim-
        ension at least 2, with the first dimension indexing
        the training examples.
    $k$ (`int`): The number of folds on which to cross-validate.
        Default: `10`.
    $bail_after$ (`int`): The number of steps of cross_validation
        training after which to bail if no improvement is seen.
        Default: `10`.

  Kwargs:
    $valid_crit$ (`nn.Module`): The validation criterion to use
        when gauging the accuracy of the model on test data. If
        `None`, this is set to `crit`; i.e., the training criter-
        ion. Default: `None`.
    $cent_norm_feats$ (`Tuple[bool]`): Tuple with first entry det-
        ermining whether to center the features; the second en-
        try determines whether to normalize the features. De-
        fault: `(True, True)`.
    $cent_norm_targs$ (`Tuple[bool]`): Tuple with first entry det-
        ermining whether to center the targets, and the second,
        whether to normalize them. Default: `(True, True)`.
    $learn_params$
        (`Union[dict,LearnParams_,torch.optim.Optimizer]`):
        The training (or 'learning') hyper-parameters in the
        form of an instance of the class `LearnParams_`; or, for
        basic functionality, a `dict` whose keys map the string
        'lr', and optionally 'mo', to `float`s; or an instance of
        `torch.optim.Optimizer`. Default: `{'lr':0.1}`.
    $bs$ (`int`): The mini-batch size where -1 forces batch gradi-
        ent descent (i.e. feed-forwarding all training examples
        before each backpropagation). Default: `-1`.
    $epochs$ (`int`): The number of epochs to train over for each
        validation step. Default: `1`.
    $verb$ (`int`): The verbosity. 0: silent, or 1. Default: `1`.
    $gpu$ (`int`): Which gpu to use in the presence of one or more
        gpus, where -1 means to use the last gpu found, and -2
        means to override using a found gpu and use the cpu.
        Default: `-1`.

  Returns:
    `(nn.Module, float)`. The trained 'best' `model` along with the
        average of that model's `k` validations.
  """
  #_this is cross_validate
  du.utils._check_kwargs(kwargs,['k','bail_after','valid_crit',\
      'cent_norm_feats','cent_norm_targs','learn_params','bs',\
      'epochs','verb','gpu'])
  valid_crit = kwargs.get('valid_crit', None)
  k = kwargs.get('k', 10)
  bail_after = kwargs.get('bail_after', 5)
  assert 2 <= len(train_data) <= 3, dedent("""\
      Argument train_data tuple must have length 2 or 3, not {}
  """.format(len(train_data)))
  feats_lengths = None
  if len(train_data) == 2:
    feats, targs = train_data
  elif len(train_data) == 3:
    feats, feats_lengths, targs = train_data
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  assert not feats_lengths, 'variable length not implemented yet'
  cent_norm_feats = kwargs.get('cent_norm_feats',(True, True))
  cent_norm_targs = kwargs.get('cent_norm_targs',(True, True))
  learn_params = kwargs.get('learn_params', {'lr': 0.1})
  bs = kwargs.get('bs', -1); epochs = kwargs.get('epochs', 1)
  verb = kwargs.get('verb', 1); gpu = kwargs.get('gpu', -1)

  no_improvement = 0
  best_valids = 1e15*torch.ones(k)
  total_epochs = 0

  if len(feats) % k != 0:
    chunklength = len(feats) // k
    print("warning: the first",k-1,"chunks have size",chunklength,\
        "but the last one has size",str(len(feats) % chunklength)+".")

  if not valid_crit: valid_crit = crit

  while no_improvement < bail_after:

    model, valids = cross_validate_train(
        model = model,
        crit = crit,
        train_data = train_data,
        k = k,
        valid_crit = valid_crit,
        cent_norm_feats = cent_norm_feats,
        cent_norm_targs = cent_norm_targs,
        epochs = epochs,
        learn_params = learn_params,
        bs = bs,
        verb = verb,
        gpu = gpu)

    total_epochs += k*epochs

    if valids.mean().item() < best_valids.mean().item():
      best_model = copy.deepcopy(model)
      best_valids = valids
      no_improvement = 0
    else:
      no_improvement += 1

    if valids.mean().item() == 0.0: no_improvement = bail_after

    if verb > 0:
      print("epoch {3}; valids: mean={0:<7g} std={1:<7g}; best={2:<7g}".\
          format(valids.mean().item(),valids.std().item(),best_valids.mean().\
          item(),total_epochs)+' '+str(no_improvement)+"/"+str(bail_after))

  if verb > 0:
    print("best valid:  mean={0:.5g}  stdev={1:.5g}".\
        format(best_valids.mean().item(),best_valids.std().item()))

  return best_model, best_valids.mean()

def optimize_ols(feats, **kwargs):
  """Compute the optimal learning rate and, optionally, momen-
  tum.

  The returned values are only optimal (or even relevant) for
  linear regression models; i.e. for linear models with MSE
  loss.

  Consider setting the verbosity to 1 so as to see the reports
  on the following during opitmization:
    - The condition number of A = X^T*X where X is the design
      matrix.
    - Check for sparseness of A when appropriate.

  Args:
    $feats$ (`torch.Tensor`): The features of the training data.

  Kwargs:
    $with_mo$ (`bool`): Optimize both the learning rate and the
        momentum. Default: `True`.
    $verb$ (`int`): Verbosity; 0 for silent, 1 to print details
        of the optimization process including warnings concern-
        ing numerical integrity. Put 2, to actually print out
        X^T*X. Default: `0`.

  Returns:
    `dict`: A dictionary mapping either 'lr' to a float or, if
        `with_mo` is `True`, so mapping both 'lr' and 'mo'.
  """
  du.utils._check_kwargs(kwargs,['with_mo','verb'])

  #from scipy.linalg import eigh
  from scipy.sparse.linalg import eigsh
  from scipy.sparse import issparse

  with_mo = kwargs.get('with_mo', True)
  verb = kwargs.get('verb', 0)

  problematic = False
  if verb: print("optimizing:")

  feats = torch.cat((torch.ones(len(feats),1), feats.to("cpu")), 1)

  design_mat = feats.transpose(0,1).mm(feats)
  if verb > 1: print(design_mat)
  eigs, _ = torch.symeig(design_mat)
  if not all(map(lambda x: x >= 0.0, eigs.tolist())):
    if verb:
      print('  warning: negative eigenvalues (most negative is {:.3g})'.\
          format(min([x for x in eigs])))
    problematic = True

  if problematic:
    from importlib.utils import find_spec
    spec = find_spec('scipy.sparse')
    if spec is None:
      if verb: print('  warning: scipy.sparse not installed.')
    else:
      from scipy.sparse.linalg import eigsh
      from scipy.sparse import issparse
      if verb: print("  checking for sparseness ... ",end='')
      feats = feats.numpy().astype('float64')
      design_mat = feats.transpose() @ feats
      is_sparse = issparse(design_mat)
      if verb: print(is_sparse)
      largest = eigsh(design_mat,1,which='LM',return_eigenvectors=False).item()
      smallest=eigsh(design_mat,1,which='SA',return_eigenvectors=False,
          sigma=1.0).item()
  else:
    eigs = eigs.tolist()
    eigs_ = [0.0 if x < 0.0 else x for x in eigs]
    if len(eigs_) < len(eigs) and verb:
      print('lopped off non-positive eig. vals.')
    largest = max(eigs_)
    smallest = min(eigs_)

  if (smallest != 0):
    if verb: print("condition number: {:.3g}".format(largest/smallest))
  else:
    if verb: print("condition number: infinite")

  if not with_mo:
    learning_rate = 2/(smallest + largest)
    momentum = 0.0
  else:
    learning_rate = (2/(smallest**0.5+largest**0.5))**2
    momentum = ((largest**0.5-smallest**0.5)/(largest**0.5+smallest**0.5))**2

  if with_mo:
    return_dict = {'lr': learning_rate, 'mo': momentum}
  else:
    return_dict = {'lr': learning_rate}

  return return_dict

def confusion_matrix(prob_dists, yss, classes=None, **kwargs):
  """Compute and optionally display the confusion matrix.

  Compute the confusion matrix with respect to given `prob_dists`
  and targets, `yss`. The columns in the displayed table corres-
  pond to the actual (i.e,, correct) target class; the rows are
  the class predicted by model.

  Note: the `gpu` argument is ignored unless `prob_dists` has the
  form of the model tupled with the inputs (see below). In that
  case, the model as well as the inputs are moved in place to
  the device determined by `gpu`.

  Args:
    $prob_dists$ (`torch.Tensor`): A tensor of dimension 2 holding,
        for each example, the probability distribution predict-
        ing the correct class. The first dimension must index
        the examples. This argument is, then, the predictions,
        in the form of probability distributions, made by a mo-
        del when fed the features of some set of examples. This
        should often be just `model(xss)`, for example.
        Alternatively, `prob_dists` can be a tuple consisting of,
        first, `model` and, second, the inputs `xss` of some data;
        in this case, the model will be applied to the data on
        the devie determined by `gpu`.
    $yss$ (`IntTensor`): A 1-dimensional tensor holding the cor-
        rect class (as some flavor of an `int`) for each example.
  Kwargs:
    $classes$ (`IntTensor`): A 1-dimensional tensor holding the nu-
        merical classes. This is naturally `torch.arange(10)` for
        digit classification, for instance.
        Default: `torch.arange(len(prob_dists[0]))`.
    $return_error$ (`bool`): If `True`, return the error in the form
        of a `float` between 0 and 1, inclusive; if `False`, return
        a `float` representing the proportion of examples correc-
        tly classified. Default: `False`.
    $show$ (`bool`): If `True`, display the (ascii) confusion matrix.
        Default: `False`.
    $class2name$ (`Dict[int, str]`): A dictionary mapping each num-
        erical class to its classname. Default: `None`.
    $gpu$ (`Union[torch.device, int]`): The GPU to use if there are
        any available. Set this to -1 to use the last GPU found
        or, if none GPUs are found, use the (first) CPU; set to
        -2 to override using any found GPU and instead use the
        CPU. Alternatively, one can set this to an instance of
        `torch.device`. Default: `-1`.

  Returns:
    `float`. The total proportion of correct predictions or, opt-
        ionally, one minus that ratio (i.e., the error).
  """
  #check and get kwargs
  du.utils._check_kwargs(kwargs,['classes','return_error','show','class2name',
      'gpu'])
  classes = kwargs.get('classes', None)
  return_error = kwargs.get('return_error', False)
  show = kwargs.get('show', False)
  class2name = kwargs.get('class2name', None)
  gpu = kwargs.get('gpu', -1)
  device = gpu if isinstance(gpu,torch.device) else du.utils.get_device(gpu)

  # check things and if necessary push the inputs through the model
  if classes is not None:
    assert classes.dim() == 1,\
        'The classes argument should be a 1-dim tensor not a {}-dim one.'\
            .format(classes.dim())
  assert isinstance(yss, IntTensor),\
      'Argument yss must be a Long-, Int-, or ShortTensor, not {}.'.\
          format(yss.type() if isinstance(yss,torch.Tensor) else type(yss))
  if not isinstance(prob_dists, torch.Tensor):
    assert (isinstance(prob_dists, tuple) or isinstance(yhatss, list)),\
        'Argument prob_dists must be a tuple like (model, xss) or a list'
    assert (isinstance(prob_dists[0], nn.Module) and\
        isinstance(prob_dists[1], torch.Tensor)), dedent("""\
            If agrument prob_dists is an interable, then the first item
            should be the model, and the second should be the tensor xss.""")
    model = prob_dists[0].to(device)
    with torch.no_grad():
      prob_dists = model(prob_dists[1].to(device))
  if classes is None:
    classes = torch.arange(len(prob_dists[0]))
  assert len(prob_dists) == len(yss),\
      'Number of features ({}) must equal number of targets ({}).'\
          .format(len(prob_dists), len(yss))
  assert prob_dists.dim() == 2,\
      'The prob_dists argument should be a 2-dim tensor not a {}-dim one.'\
          .format(prob_dists.dim())

  # compute the entries in the confusion matrix
  cm_counts = torch.zeros(len(classes), len(classes))
  for prob, ys in zip(prob_dists, yss):
    cm_counts[torch.argmax(prob).item(), ys] += 1
  cm_pcts = cm_counts/len(yss)
  counts = torch.bincount(yss, minlength=len(classes))

  # display the confusion matrix
  if show:
    cell_length = 5
    print(((cell_length*len(classes))//2+1)*' '+"Actual")
    print('     ',end='')
    for class_ in classes:
      print('{:{width}}'.format(class_.item(), width=cell_length),end='')
    if class2name: print(' '*len(list(class2name.values())[0])+'   (correct)')
    else: print(' (correct)')
    print('     '+'-'*cell_length*len(classes))
    for i, row in enumerate(cm_pcts):
      print(str(i).rjust(3),end=' |')
      for j, entry in enumerate(row):
        if entry == 0.0:
          print((cell_length-1)*' '+'0', end='')
        elif entry == 100.0:
          print((cell_length-3)*' '+'100', end='')
        else:
          string = '{:.1f}'.format(100*entry).lstrip('0')
          length = len(string)
          if i==j:
            string = du.utils._markup('~'+string+'~')
          print(' '*(cell_length-length)+string, end='')
      n_examples = cm_counts[:,i].sum()
      pct = 100*(cm_counts[i,i]/n_examples) if n_examples != 0 else 0
      if class2name:
        print('  {} ({:.1f}% of {})'.format(class2name[i],pct,int(counts[i])))
      else:
        print(' ({:.1f}% of {})'.format(pct, int(counts[i])))

  if return_error:
    return 1-torch.trace(cm_pcts).item()
  else:
    return torch.trace(cm_pcts).item()

def r_squared(yhatss, yss, **kwargs):
  """Compute r_squared.

  Returns the coefficient of determination of two 2-d tensors
  (where the first dimension in each indexes the examples), one
  holding the `yhatss` (the predicted outputs) and the other hol-
  ding the actual outputs, `yss`.

  Note: `yhatss` and `yss` - or `model`, `xss`, and `yss` if the second
  option (see below) is used - are each moved in place to the
  to the device determined by `gpu`.

  Args:
    $yhatss$ (`torch.Tensor`): Either the predicted outputs (assum-
        ed to be of shape `(len(yhatss), 1)` (which is often just
        `model(xss)`) or a tuple of the form `(model, xss)`; use
        the second option to move both `model` and `xss` to the de-
        vice determined by `gpu` before computing `model`(`xss`).
    $yss$ (`torch.Tensor`): The actual outputs.

  Kwargs:
    $return_error$ (`bool`): If `False`, return the proportion of the
        variation explained by the regression line. If `True`,
        return 1 minus that proportion. Default: `False`.
    $gpu$ (`Union[torch.device, int]`): The GPU to use if there are
        any available. Set this to -1 to use the last GPU found
        or, if none GPUs are found, use the (first) CPU; set to
        -2 to override using any found GPU and instead use the
        CPU. Alternatively, one can set this to an instance of
        `torch.device`. Default: `-1`.

  Returns:
    `float`. The proportion of variation explained by the model
        (as compared to a constant model) or (optionally) 1 mi-
        nus that proportion (i.e., the proportion unexplained).

  >>> `yhatss = torch.arange(4.).unsqueeze(1)`
  >>> `yss = torch.tensor([-1., 5., 2., 3.]).unsqueeze(1)`
  >>> `r_squared(yhatss, yss)`
  0.09333...
  """
  du.utils._check_kwargs(kwargs,['return_error','gpu'])
  return_error = kwargs.get('return_error', False)
  gpu = kwargs.get('gpu', -1)
  device = gpu if isinstance(gpu, torch.device) else du.utils.get_device(gpu)

  if not isinstance(yhatss, torch.Tensor):
    assert isinstance(yhatss, (tuple,list)),\
        'Argument yhatss must be a tuple of the form (model, tensor), or a list'
    assert (isinstance(yhatss[0], nn.Module) and\
        isinstance(yhatss[1], torch.Tensor)), dedent("""\
            If agrument yhatss is an iterable, then the first item should be
            the model, and the second should be the xss.""")
    model = yhatss[0].to(device)
    with torch.no_grad():
      yhatss = model(yhatss[1].to(device))
  assert yhatss.dim() == yss.dim(), dedent("""\
      The arguments yhatss (dim = {}) and yss (dim = {}) must have the
      same dimension.""".format(yhatss.dim(), yss.dim()))
  assert yhatss.dim() == 2, dedent("""\
      Multiple outputs not implemented yet; yhatss should have dimen-
      sion 2, not {}.""".format(yhatss.dim()))
  assert len(yhatss) == len(yss), dedent("""\
      len(yhatss) is {} which is not equal to len(yss) which is {}
  """.format(len(yhatss),len(yss)))
  assert yhatss.size()[1] ==  yss.size()[1] == 1, dedent("""\
      The first dimension of yhatss and yss should index the examples.""")
  ave_sum_squares = nn.MSELoss()
  yhatss = yhatss.squeeze(1).to(device)
  yss = yss.squeeze(1).to(device)
  SS_E = len(yss) * ave_sum_squares(yhatss, yss)
  SS_T=len(yss)*ave_sum_squares(yss,yss.mean(0)*torch.ones(len(yss)).to(device))
  if return_error: return (SS_E/SS_T).item()
  else: return 1.0-(SS_E/SS_T).item()

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
