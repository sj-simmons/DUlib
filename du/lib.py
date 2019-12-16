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
     $*args$)      -each of these tensors are split into two

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
     $print_lines$ = `(17,7)`,
                 -print 17 beginning lines and 7 ending lines
     $verb$ = `2`,   -verbosity; 3 for more, 1 for less, 0 silent
     $gpu$ = `-1`)   -the gpu to run on, if any are available; if
                  none available, use the cpu; put -1 to use
                  the last gpu if multiple ones found; put -2
                  to override found gpu(s) and use the cpu.
                  Consider just accepting the default here.

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
#  - Looks like the graphing epochs values on axis are off by one
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
#  - Add percentage or loss to ascii output in the presence of
#    testing data. (Take into account forget_after here).
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

import torch
import torch.nn as nn
from types import FunctionType
from typing import Dict
from textwrap import dedent
import du.util

__author__ = 'Scott Simmons'
__version__ = '0.9'
__status__ = 'Development'
__date__ = '12/23/19'
__copyright__ = """
  Copyright 2019 Scott Simmons

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

def center(xss, new_centers = None):
  """Center a tensor.

  With this you can translate data to anywhere. If the second
  argument is `None`, then this simply mean-centers the data al-
  ong the first dimension; in other words, it rigidly translat-
  es `xss` so that its mean along the first dimension is the zero
  tensor.

  Notice that the returned object is a tuple. So if you want to
  simply mean-center a tensor you would call this function like
               `xss_centered, _ = center(xss)`
  That is, you can use an underscore (or whatever) if you don't
  need the second element of the tuple being returned.

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

  >>> `xss = torch.arange(12.).view(3,4)`
  >>> `center(xss)`
  (tensor([[-4., -4., -4., -4.],
          [ 0.,  0.,  0.,  0.],
          [ 4.,  4.,  4.,  4.]]), tensor([4., 5., 6., 7.]))
  >>> `xss_, xss_means =  center(xss)`
  >>> `xss__, _ = center(xss_, -xss_means)`
  >>> `int(torch.all(torch.eq(xss, xss__)).item())`
  1
  """
  # add an assert checkin that new_center is right size.
  xss_means = xss.mean(0)
  if isinstance(new_centers, torch.Tensor):
    new_xss = xss.sub_(new_centers)
  else:
    new_xss = xss.sub_(xss_means)
  return new_xss, xss_means

def normalize(xss, new_widths = None, unbiased = True):
  """Normalize without dividing by zero.

  See the documentation for the function `center`. This is com-
  pletely analagous.

  Args:
    $xss$ (`torch.Tensor`)
    $new_widths$ (`torch.Tensor`)
    $unbiased$ (`bool`): If unbiased is `False`, divide by `n` instead
        of `n-1` when computing the standard deviation.

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

def coh_split(prop, *args):
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
  assert 0 <= prop <= 1, dedent("""\
      Arg prop ({}) must be between 0 and 1, inclusive.
  """.format(prop))
  len_ = list(map(len, args))
  assert all(len_[0] == x for x in len_), "all tensors must have same size "+\
      "in first dim"
  indices = torch.randperm(len_[0])
  rand_args = [tensor.index_select(0, indices) for tensor in args]
  cutoff = int(prop * len_[0])
  split_args = [[tensor[:cutoff], tensor[cutoff:]] for tensor in rand_args]
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
    return 'learning rate: ' + du.util.format_num(self.lr)

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
    return super().__str__() + ', momentum: ' + du.util.format_num(self.mo)

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

def train(model, crit, train_data, **kwargs):
  """Train a model.

  Assuming that the number of training examples is divisible by
  the batchsize, the loss printed is the average loss per sam-
  ple over each epoch training. (Under the same assumption,
  one epoch of training corresponds to the model seeing each
  example in the training data exactly once.

  If graphing is enabled, the black graph is the (average)
  loss for the training data. If testing data is present,
  then the red graph is the average loss per example for the
  test data.

  Notes on specifying training hyper-parameters:

  The argument `learn_params` specifies the training hyper-
  parameters.  It can be used in one of two ways. To train with
  a constant learning rate and/or momentum, one passes a sim-
  ple dictionary of the form either

       `train( ..., learn_params = {'lr': 0.01}, ...)`

  or

    `train( ..., learn_params = {'lr': 0.01, 'mo': 0.9}, ...).`

  Alternatively, `learn_params` can be an instance of (a subclass
  of) the `LearnParams_` class or an instance of `torch.optim.Opti`
  `mizer`.

  Args:
    $model$ (`nn.Module`): The instance of Module to be trained.
    $crit$ (`nn.modules.loss`): The loss function when training.
    $train_data$ (`Tuple[torch.Tensor]`): A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tensor is
        only necessary when training a recurrent net on var-
        iable length inputs. In that case, the triple of ten-
        sors must be of the form
          `(train_features, train_lengths, train_targets)`.
        That is, the first tensor holds the inputs of the
        training data, the second holds the corresponding
        lengths, and the third holds the training data out-
        puts.
        If the data are not of variable length, then there
        is no need to pass the middle tensor in the triple
        above. So one passes
                 `(train_features, train_targets)`.
        In any case, each of the tensors in the tuple must
        be of dimension at least two, with the first dimen-
        sion indexing the training examples.

  Kwargs:
    $test_data$ (`Tuple[torch.Tensor]`): Data to test on in the
        form of a tuple of length 2 or 3 (that is, matching the
        `train_data` (see above).  The loss on test data is com-
        puted each epoch.  However, The test data is not shown
        to the model as part of backpropagation. Default: `None`.
    $learn_params$
        (`Union[dict,LearnParam_,torch.optim.Optimizer]`):
        The training (or 'learning') hyperparameters in the form
        of an instance of a subclass of `LearnParams_`; or, for
        basic functionality, a `dict` that maps the string 'lr',
        and optionally 'mo', to `float`s; or an instance of `torch`
        `.optim.Optimizer`. Default: `{'lr':0.1}`.
    $bs$ (`int`): The mini-batch size where -1 forces batch grad-
        ient descent (i.e. feed-forwarding all training exam-
        ples before each backpropagation). Default: -1.
    $epochs$ (`int`): The number of epochs to train over, where
        an epoch is duration required to see each training ex-
        ample exactly once. Default: 10.
    $graph$ (`int`): If positive then, during training, display
        a real-time graph. If greater than 1, then the be-
        gining `graph` losses are thrown away when training
        gets to epoch `graph` (this functionality is made
        available for a better viewing experience for some
        models). Requires matplotlib (and a running X server).
        If 0, do not display a graph. Default: 0.
    $print_lines$ (`Tuple[int, int]`): A tuple, the first compon-
        ent of which is the number of lines to print initial-
        ly when printing the current loss for each epoch dur-
        ing training, and the second of which is the number
        of lines to print lastly when training. If at least
        one element of the tuple is 0 (resp., -1), then no
        (resp., all) lines are printed. Default: (17, 7).
    $verb$ (`int`): The verbosity. 0: silent, ... , 3: all.
        Default: 2.
    $gpu$ (`int`): The gpu to use if there are any available. Set
        to -1 to use the last gpu found when gpus are present;
        set to -2 to override using a found gpu and use the
        cpu. Default -1.

  Returns:
    `nn.Module`. The trained model sent to device 'cpu'.
  """
  # this is train
  du.util._check_kwargs(kwargs,['test_data','learn_params','bs','epochs',
      'graph','print_lines','verb','gpu'])
  du.util._catch_sigint()

  test_data = kwargs.get('test_data', None)
  learn_params = kwargs.get('learn_params', {'lr': 0.1})
  bs = kwargs.get('bs', -1); epochs = kwargs.get('epochs', 10)
  print_init, print_last = kwargs.get('print_lines',(8, 12))
  verb = kwargs.get('verb', 2); graph = kwargs.get('graph', 0)
  gpu = kwargs.get('gpu', -1)

  assert graph>=0, 'graph must be a non-negative integer, not {}.'.format(graph)

  device = du.util.get_device(gpu)
  train_feats, train_feats_lengths, train_targs =\
      du.util._parse_data(train_data, device)
  num_examples = len(train_feats)

  if bs <= 0: bs = num_examples

  if verb > 0: print('training on', device)

  model = model.to(device)
  if verb > 2: print(model)

  #process learn_params
  has_optim = False
  if isinstance(learn_params, Dict):
    for key in learn_params.keys(): assert key in ['lr','mo'],\
        "keys of learn_params dict should be 'lr' or 'mo', not {}.".format(key)
    assert 'lr' in learn_params.keys(), "input dict must map 'lr' to float"
    lr = learn_params['lr']
    if verb > 1: print('learning rate:', du.util.format_num(lr), end=', ')
    if 'mo' not in learn_params.keys():
      learn_params = LearnParams_(lr = lr)
      mo = None
    else:
      mo = learn_params['mo']
      if verb > 1: print('momentum:', du.util.format_num(mo), end=', ')
      learn_params = Momentum(model, lr = lr, mo = mo)
      learn_params.set_device(device)
    if verb > 1: print('batchsize:', bs)
  elif isinstance(learn_params, torch.optim.Optimizer):
    has_optim = True
  else:
    assert isinstance(learn_params, LearnParams_), dedent("""\
        learn_params must be a dict or an instance of a subclass of
        LearnParams_, not a {}.
    """.format(type(learn_params)))
    learn_params.set_device(device)
    if verb > 1: print(learn_params, end=', ')
    if verb > 1: print('batchsize:', bs)

  if test_data:
    if len(test_data[0]) == 0: test_data = None
    else:
      test_feats, test_feats_lengths, test_targs =\
          du.util._parse_data(test_data, device)
      losses_test=[]

  if print_init == -1 or print_last == -1: print_init, print_last = epochs, -1

  if graph:
    import matplotlib.pyplot as plt
    plt.ion(); fig, _ = plt.subplots()
    plt.xlabel('epoch', size='larger'); plt.ylabel('average loss',size='larger')

  losses = []

  for epoch in range(epochs):
    accum_loss = 0
    indices = torch.randperm(len(train_feats)).to(device)

    for idx in range(0, num_examples, bs):
      current_indices = indices[idx: idx + bs]

      if isinstance(train_feats_lengths, torch.Tensor):
        loss = crit(
            model(
                train_feats.index_select(0, current_indices),
                train_feats_lengths.index_select(0, current_indices)),
            train_targs.index_select(0, current_indices))
      else:
        loss = crit(
            model(train_feats.index_select(0, current_indices)),
            train_targs.index_select(0, current_indices))

      accum_loss += loss.item()
      if has_optim: learn_params.zero_grad()
      else: model.zero_grad()
      loss.backward()

      if has_optim: learn_params.step()
      else: learn_params.update(model.parameters())

    if print_init * print_last != 0 and verb > 0:
      loss_len = 20
      base_str = "epoch {0}/{1}; loss ".format(epoch+1, epochs)
      loss_str = "{0:<10g}".format(accum_loss*bs/num_examples)
      if epochs < 20 or epoch < print_init:
        print(base_str + loss_str)
      elif epoch > epochs - print_last:
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
      if isinstance(train_feats_lengths, torch.Tensor):
        loss = crit(model(train_feats, train_feats_lengths), train_targs).item()
      else:
        loss = crit(model(train_feats), train_targs).item()
      losses.append(loss)
      if test_data:
        if isinstance(test_feats_lengths, torch.Tensor):
          loss = crit(model(test_feats, test_feats_lengths), test_targs).item()
        else:
          loss = crit(model(test_feats), test_targs).item()
        losses_test.append(loss)
      if epoch == graph:
        losses = losses[graph:]
        if test_data: losses_test = losses_test[graph:]
        plt.clf()
        plt.xlabel('epoch',size='larger');
        plt.ylabel('average loss',size='larger')
      plt.plot(range(graph,graph+len(losses)), losses, c='black', lw=.8);
      if test_data:
        plt.plot(range(graph,graph+len(losses_test)),losses_test,c='red',lw=.8);
      fig.canvas.flush_events()

  if graph:
    plt.plot(range(graph,graph+len(losses)),losses,c='black',lw=.8,\
        label='training')
    if test_data: plt.plot(range(graph,graph+len(losses_test)),\
        losses_test,c='red',lw=.8,label='testing')
    plt.legend(loc=1); plt.ioff(); plt.show()

  model = model.to('cpu')
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
        Default: 10.

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
  # This is cross_validate_train
  du.util._check_kwargs(kwargs,['k','valid_crit','cent_norm_feats',\
      'cent_norm_targs','learn_params','bs','epochs','gpu','verb'])
  du.util._catch_sigint()
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
        Default: 10.
    $bail_after$ (`int`): The number of steps of cross_validation
        training after which to bail if no improvement is seen.
        Default: 10.

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
  # This is cross_validate
  du.util._check_kwargs(kwargs,['k','bail_after','valid_crit',\
      'cent_norm_feats','cent_norm_targs','learn_params','bs',\
      'epochs','verb','gpu'])
  import copy
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
    from importlib.util import find_spec
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

def confusion_matrix(prob_dists, yss, classes, **kwargs):
  """Compute the confusion matrix.

  Compute the confusion matrix with respect to given `prob_dists`
  and targets, `yss`. The columns in the displayed table corres-
  pond to the actual (correct) target class; the rows are the
  class predicted by model.

  Args:
    $prob_dists$ (`torch.Tensor`): A tensor of dimension 2 holding,
        for each example, the probability distribution predict-
        ing the correct class. The first dimension must index
        the examples. This argument is, then, the predictions,
        in the form of probability distributions, made by a mo-
        del when fed the features of some set of examples. This
        should often be just `model(xss)`, for example.
    $yss$ (`torch.LongTensor`): A 1-dimensional tensor holding the
        correct class for each example.
    $classes$ (`torch.LongTensor`): A 1-dimensional tensor holding
        the numerical classes. This should be `torch.arange(10)`
        for digit classification, for instance.

  Kwargs:
    $return_error$ (`bool`): If `True` return the error in the form
        of a `float` between 0 and 1, inclusive, representing
        the error; if `False`, return a float representing the
        proportion of examples correctly classified. Default:
        `False`.
    $show$ (`bool`): If `True` display the (ascii) confusion mat-
        rix. Default: `False`.
    $class2name$ (`Dict[int, str]`): A dictionary mapping each num-
        erical class to its classname. Default: `None`.

  Returns:
    `float`. The total proportion (a number between 0 and 1) of
        correct predictions or (optionally) one minus that rat-
        io (i.e., the error rate).
  """
  # this is confusion_matrix
  du.util._check_kwargs(kwargs,['return_error','show','class2name'])
  assert len(prob_dists) == len(yss),\
      'Number of features ({}) must equal number of targets ({}).'\
          .format(len(prob_dists), len(yss))
  assert prob_dists.dim() == 2,\
      'The prob_dists argument should be a 2-dim tensor not a {}-dim one.'\
          .format(prob_dists.dim())
  assert classes.dim() == 1,\
      'The classes argument should be a 1-dim tensor not a {}-dim one.'\
          .format(classes.dim())
  assert isinstance(yss,torch.LongTensor), 'Argument yss must be a LongTensor.'
  return_error = kwargs.get('return_error', False)
  show = kwargs.get('show', False)
  class2name = kwargs.get('class2name', None)

  cm_counts = torch.zeros(len(classes), len(classes))
  for prob, ys in zip(prob_dists, yss):
    cm_counts[torch.argmax(prob).item(), ys] += 1

  cm_pcts = cm_counts/len(yss)
  counts = torch.bincount(yss, minlength=len(classes))

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
            string = du.util._markup('~'+string+'~')
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
  (where the first dimension in each indexes the examples),
  one holding the yhatss (the predicted outputs) and the other
  holding the actual outputs, yss.

  Args:
    $yhatss$ (torch.Tensor): Either the predicted outputs (assum-
        ed to be of shape `(len(yhatss), 1)` (which is often sim-
        ply `model(xss)` or a tuple of the form `(model, xss)` (use
        this second option if you want the `xss` pushed through
        `model` on the fastest available device).
    $yss$ (`torch.Tensor`): The actual outputs.

  Kwargs:
    $return_error$ (`bool`): If False return the proportion of the
        variation explained by the regression line. If True,
        return 1 minus that proportion. Default: False.
    $gpu$ (`int`): The gpu to use if there are any available. Set
        to -1 to use the last gpu found when gpus are present;
        set to -2 to override using a found gpu and use the
        cpu. Default -1.

  Returns:
    `float`. The proportion of variation explained by the model
        (as compared to a constant model) or (optionally) 1 mi-
        that, i.e., the proportion unexplained.

  >>> yhatss = torch.arange(4.).unsqueeze(1)
  >>> yss = torch.tensor([-1., 5., 2., 3.]).unsqueeze(1)
  >>> r_squared(yhatss, yss)
  0.09333...
  """
  du.util._check_kwargs(kwargs,['return_error','gpu'])
  return_error = kwargs.get('return_error', False)
  gpu = kwargs.get('gpu', -1)
  device = du.util.get_device(gpu)
  if not isinstance(yhatss, torch.Tensor):
    assert (isinstance(yhatss, tuple) or isinstance(yhatss, list)),\
        'Argument yhatss must be a tuple or a list'
    assert (isinstance(yhatss[0], nn.Module) and\
        isinstance(yhatss[1], torch.Tensor)), dedent("""\
            If agrument yhatss is an interable, then the first item
            should be the model, and the second should be the xss
        """)
    model = yhatss[0].to(device)
    yhatss = model(yhatss[1].to(device))
  assert yhatss.dim() == yss.dim(), dedent("""\
      The arguments yhatss (dim = {}) and yss (dim = {}) must have
      the same dimension.
  """.format(yhatss.dim(), yss.dim()))
  assert yhatss.dim() == 2, dedent("""\
      Multiple outputs not implemented yet; yhatss should have dim-
      ension 2, not {}.
  """.format(yhatss.dim()))
  assert len(yhatss) == len(yss), dedent("""\
      len(yhatss) is {} which is not equal to len(yss) which is {}
  """.format(len(yhatss),len(yss)))
  assert yhatss.size()[1] ==  yss.size()[1] == 1, dedent("""\
      The first dimension of yhatss and yss should index the examples.
  """)
  ave_sum_squares = nn.MSELoss()
  yhatss = yhatss.squeeze(1).to(device)
  yss = yss.squeeze(1).to(device)
  SS_E = len(yss) * ave_sum_squares(yhatss, yss)
  SS_T = len(yss) * ave_sum_squares(yss, yss.mean(0) *\
      torch.ones(len(yss)).to(device))
  if return_error:
    return (SS_E/SS_T).item()
  else:
    return 1.0-(SS_E/SS_T).item()

if __name__ == '__main__':
  import inspect
  import doctest

  # find the user defined functions
  _local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
       if callable(ob) and ob.__module__ == __name__]

  #remove markdown
  #  from the docstring for this module
  globals()['__doc__'] = du.util._markup(globals()['__doc__'],strip = True)
  #  from the functions (methods are fns in Python3) defined in this module
  for _, _ob in _local_functions:
    if inspect.isfunction(_ob):
      _ob.__doc__ = du.util._markup(_ob.__doc__,strip = True)
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
        _ob.__dict__[name].__doc__ = du.util._markup(_meth.__doc__,strip = True)

  # run doctests
  failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)

  # print signatures
  if failures == 0:
    from inspect import signature
    for name, ob in _local_functions:
      print(name,'\n  ', inspect.signature(ob))
