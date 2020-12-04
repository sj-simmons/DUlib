#!/usr/bin/env python3
"""core functionality for working with neural nets.

This library can be used to center and normalize data, split
out testing data, train neural nets, and gauge performance of
trained models.

`QUICK SIGNATURES`

  ~data related tools:~

  |coh_split|    randomize and coherently split each tensor in
               `*args`; returns `Tuple[tensor]`
    ($prop$,       -split like `prop`, 1-`prop`
     $*args$,      -each of these tensors are split into two
     $randomize$ = `True`)
                 -whether to randomize before splitting

  |split_df|     split a dataframe into disjoint subframes; ret-
               urns `Tuple[dataframe]`
    ($df$,         -the dataframe to be split
     $splits$)     -a tuple of positive floats which sum to 1
                  or less

  |center|       mean-center `xss`; returns `(tensor, tensor)`
    ($xss$,        -tensor to center w/r to its 1st dimension
     $shift_by$ = `None`)
                 -the first returned tensor is `xss` with its
                  columns shifted according to `shift_by`; the
                  default leads to `shift_by` = `xss.means(0)`;
                  i.e. mean-centering `xss`.

  |normalize|    normalize `xss`; returns `(tensor, tensor)`
    ($xss$,        -tensor to normalize w/r to its 1st dimension
     $scale_by$ = `None`,
                 -first tensor returned will now have columns
                  scaled according to `scale_by`; default leads
                  to dividing each entry in a column by that
                  columns st. dev. but leaving unchanged any
                  column with st. deviation close to 0.
     $unbiased$ = `True`,
                 -use n-1 instead of n in the denominator when
                  computing the standard deviation
     $threshold$ = `1e-6`)
                 -do not divide by a number smaller than this

  |standardize|  standardize data; returns `tensor`
    ($xss$,      -the data to be standardized, where the first
                  dimension indexes the examples
     $means$ = `None`,
                 -subtract these means, columnwise, from `xss`
     $stdevs$ = `None`,
                 -divide by these, columnwise, but do not div-
                  ide by zero
     $threshold$ = `1e-6`)
                 -do not divide by a number smaller than this

  |online_means_stdevs|
                 -compute the means and stdevs of large or aug-
                  mented datasets; returns a tuple of tupled
                  pairs of tensors.
    ($data$,       -a pair `(xss, yss)` of tensors or a dataloader
     $transforms_$,-any number of augmentations of the data
     $batchsize=1$)-the online computation is done this many exa-
                  mples at a time.

  ~tools for training:~

  |train|        return `model` trained using SGD;
    ($model$,      -the instance of `nn.Module` to be trained
     $crit$,       -the criterion for assessing the loss
     $train_data$, -either a tuple `(train_feats, train_targs)` or
                  `(train_feats, train_lengths, train_targs)`;
                  passing `train_lengths` or, below, `test_lengths`
                  is only relevant for certain recurrent nets.
                  This can also be a dataloader yielding tuples
                  as above.
     $valid_data$ = `None`,
                 -either `(valid_feats, valid_targs)` or
                  `(valid_feats, valid_lengths, valid_targs)`;
                  or a dataloader yielding such tuples.
     $valid_metric$ = `True`,
                 -function determining how the model is valida-
                  ted w/r to `valid_data`. The default, `True`, re-
                  sults in using `explained_var` for regression
                  and `class_accuracy` for classification.
     $learn_params$ = `{'lr': 0.1}`,
                 -a `dict` of the form `{'lr': 0.1,'mo': 0.9}` or
                  `{'lr': 0.1}`, or an instance of `LearnParams_`,
                  or an instance of `torch.optim.Optimizer`.
     $bs$ = `-1`,    -the mini-batch size; -1 is (full) batch
     $epochs$ = `10`,-train for this many epochs
     $graph$ = `0`,  -put 1 or greater to show graph when training
     $print_lines$ = `(7,8)`,
                 -print 7 beginning lines and 8 ending lines;
                  put -1 to disable compressed printing.
     $verb$ = `2`,   -verbosity; 3, for verbose; 0 silent
     $gpu$ = `(-1,)`,-the gpu to run on, if any are available; if
                  none available, use the cpu; put -1 to use
                  the last gpu if multiple ones found; put -2
                  to override found gpu(s) and use the cpu.
                  Consider just accepting the default here.
     $args$ = `None`)-an instance of `argparse.Namespace` by which one
                  pass in arguments for most of the parameters
                  above. This argument is typically created us-
                  ing `du.utils.standard_args`.

  |cross_validate_train|
    ($model$, $crit$, $train_data$, $k$, $**kwargs$)
     This is a helper function for `cross_validate`; each epoch
     it iterates fold-wise, validating on the `k` possible test
     sets, and returns the partially trained (for 1 epoch, by
     default) model; consider using `cross_validate` instead of
     calling this directly; the parameters are the same as for
     `cross_validate` but without `bail_after`.

  |cross_validate| return `model` cross-validate trained tupled
                 with the mean (`float`) of `model`'s validations
    ($model$,     -the model to be cross-validated
     $crit$,      -the loss function while training
     $train_data$,-either `(train_feats, train_targs)` or
                 `(test_feats, test_lengths, train_targs)`
     $k$ = `10`,    -the number of folds when cross-validating
     $bail_after$ = `5`,
                -bail after this many steps if no improvement
     $valid_metric$ = `None`,
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

  |LearnParams_|  base class for defining learning parameters
    ($lr$ = `0.1`)  -we need at least a learning rate

  |Momentum|      subclass of `LearnParams_`, an instance of which
                adds momentum to gradient descent
    ($model$,     -model instance to which to add momentum
     $lr$ = `0.01`, -the desired learning rate
     $mo$ = `0.9`)  -the desired momentum

  |copy_parameters| helper for sub-classing `LearnParams_`
    ($model$)     -copy the parameters of `model`

  |optimize_ols|  find optimal training hyper-parameters; returns
                a dict with keys 'lr' and 'mo'
    ($feats$,     -the `xss` for the data set
     $with_mo$ = `True`
                -if `False` just returns optimal learning rate
     $verb$ = `0`)  -default is silence; put 1 to include warnings,
                 and 2 to actually print out X^T*X where X is
                 the design matrix

  ~evaluation metrics:~

  |class_accuracy| compute the proportion correct for a classifi-
                 cation problem; returns `float`
    ($model$,     -a (partially) trained model
     $data$,      -either a tuple `(xss, yss)` or a dataloader
     $classes$ = `None`,
                -a tensor of shape `(n)` holding the possible
                 classes; normally this is `torch.arange(10)`
                 if there are say 10 things being classified
     $class2name$ = `None`,
                -a dict mapping `int`s representing the classes
                 to the corresponing descriptive name (`str`)
     $show_cm$ = `False`,
                -display the confusion matrix       -
     $gpu$ = `-1`,  -run on the fastest device, by default
     $color$ = `True`)
                -whether to colorize the confusion matrix

  |explained_var| return (`float`) the explained variance
    ($model$,     -a (partially) trained model
     $data$,      -a tuple of tensors or a dataloader
     $return_error$ = `False`,
                -return the 1-explained_var if True
     $gpu$ = `-1`)  -run on the fastest device, by default
                    _____________________
"""
#Todo:
#  - consider adaptive pooling
#  - do something sensible about cudnn.benchmark
#  - get rid of batchsize = -1 stuff
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
# Done or didn't do:
#  - Add percentage or loss to ascii output in the presence of
#    testing data. DON"T DO THIS so that training without graph
#    will be fast.

import time
import math
import tkinter
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from types import FunctionType
from typing import Dict
from textwrap import dedent
import du.utils

__author__ = 'Scott Simmons'
__version__ = '0.9.3'
__status__ = 'Development'
__date__ = '12/03/20'
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

def split_df(df, splits):
  """Randomize and/or split a dataframe.

  If `splits` is `()`, return a tuple of length one consisting of
  `df` randomized. If `splits` is a nonempty tuple of proportions,
  then return a tuple (the same length as `splits`) of disjoint
  dataframes of those proportions randomly sampled from `df`.

  If the !sum! of the splits is 1, then the returned dataframes'
  union is `df`; otherwise, the returned dataframes proportion-
  ately and disjointly partition 100*!sum! percent of the data.
  E.g., if `df` consists of 500 entries and `splits=(.7,.15,.15)`,
  then the returned tuple containing dataframes of sizes 350,
  75, and 75; but, if `splits=(.4,.1,.1)`, then dataframes are
  of sizes 120, 50, and 50.

  Args:
    $df$ (`pandas.Dataframe`): The dataframe to be split.
    $splits$ (`Tuple`): A tuple of non-negative floats whose sum is
        less than or equal to 1.0, or the empty tuple `()`.

  Returns:
    `Tuple[pandas.Dataframe]`. A tuple of disjoint dataframes.

  >>> `import pandas`
  >>> `df = pandas.DataFrame({'x':list(range(10))})`
  >>> `train_df, test_df = split_df(df, (0.6,0.4))`
  >>> `print(len(train_df), len(test_df))`
  6 4
  >>> `df = pandas.DataFrame({'x':list(range(500))})`
  >>> `train_df, valid_df, test_df = split_df(df, (.7,.15,.15))`
  >>> `print(len(train_df), len(valid_df), len(test_df))`
  350 75 75
  >>> `df = pandas.DataFrame({'x':list(range(11))})`
  >>> `train_df, valid_df, test_df = split_df(df, (.4,.3,.3))`
  >>> `print(len(train_df), len(valid_df), len(test_df))`
  4 4 3
  >>> `df = pandas.DataFrame({'x':list(range(500))})`
  >>> `dfs = split_df(df, (0.4,0.1,0.1))`
  >>> `print(len(dfs[0]), len(dfs[1]), len(dfs[2]))`
  200 50 50
  >>> `df = pandas.DataFrame({'x':list(range(100))})`
  >>> `dfs = split_df(df, (0.3,0.3,0.2,0.2))`
  >>> `print(len(dfs[0]), len(dfs[1]), len(dfs[2]), len(dfs[3]))`
  30 30 20 20
  >>> `df = pandas.DataFrame({'x':list(range(100))})`
  >>> `df_ = split_df(df, ())`
  >>> `print(len(df_[0]))`
  100
  """
  assert isinstance(splits, tuple), _markup('Arg. $splits$ should be a tuple.')
  randomized = df.sample(frac = 1.0)
  returnlist = []
  if len(splits) == 0:
    returnlist.append(randomized.copy())
    return tuple(returnlist)
  else:
    sum_ = sum(splits)
    assert sum_ <= 1.0, _markup('sum of entries in arg. $splits$ must be <= 1.0')
    frac = .5; splits = [1.0] + list(splits)
    for idx in range(len(splits)-1):
      frac = (frac / (1-frac)) * (splits[idx+1] / splits[idx])
      if idx < len(splits) or sum_ < 1:
        # no real need to randomly sample here but ... whatever
        #splitout = randomized.sample(frac = 1 if frac > 1 else frac)
        #randomized = randomized.drop(splitout.index).copy()
        # nevermind, just do this
        cutoff = round(frac*len(randomized))
        splitout = randomized.head(cutoff)
        randomized = randomized.tail(-cutoff)
        returnlist.append(splitout.copy())
      else:
        returnlist.append(randomized)
  return tuple(returnlist)

def center(xss, shift_by = None):
  """Re-center data.

  With this you can rigidly translate data. The first dimen-
  sion of `xss` should be the index parameterizing the examples
  in the data.

  Suppose, for example, that we wish to translate a point-cloud
  in the plane so that it is centered at the origin:

  A randomly generated point-cloud indexed by dim 0:
  >>> `xss = torch.rand(100,2)`

  Let us now center it:
  >>> `xss, _ = center(xss)`

  And check that it is centered (at the origin):
  >>> `torch.all(torch.lt(torch.abs(xss.mean(0)),1e-5)).item()`
  1

  If `new_centers` is `None`, then `center` simply mean-centers the
  data (i.e., rigidly translates the data so that it is 'balan-
  ced' with repect to the origin):

  >>> `xss = torch.arange(2400.).view(100, 2, 3, 4)`
  >>> `means = center(xss)[0].mean(0)`
  >>> `zeros = torch.zeros(means.shape)`
  >>> `torch.all(torch.eq(means, zeros)).item()`
  1

  Notice that the returned object is a tuple. So if you want to
  simply mean-center a tensor, you would call `center` like:

                 `xss_centered, _ = center(xss)`
              or `xss_centered = center(xss)[0]`

  Args:
    $xss$ (`torch.Tensor`) The tensor to center.
    $shift_by$ (`torch.Tensor`) A tensor, the number of dimen-
        sions of which is one less than that of `xss` and whose
        shape is in fact `(d_1,`...`,d_n)` where `xss` has as its
        shape `(d_0, d_1,`...`,d_n)`. The default is `None` which is
        equivalent to `shift_by` being `xss.mean(0)`.
        The first returned tensor is `xss` with the `(i_1,`...`,i_n)`
        entry of `shift_by` subtracted from the `(j, i_1,`...`,i_n)`
	entry of `xss`, `0 <= j < d_0`.

  Returns:
    `(torch.Tensor, torch.Tensor)`. A tuple of tensors the first
        of which is `xss` shifted with respect to the first dim-
        ension according to `shift_by`, and the second of which
        is a tensor the size of the remaining dimensions holding
        the means of the original data `xss` with respect to the
        first dimension.

  More examples and tests:
  >>> `xss = torch.arange(12.).view(3,4)`
  >>> `xss`
  tensor([[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.]])
  >>> `xss_, xss_means = center(xss)`
  >>> `xss_, xss_means`
  (tensor([[-4., -4., -4., -4.],
          [ 0.,  0.,  0.,  0.],
          [ 4.,  4.,  4.,  4.]]), tensor([4., 5., 6., 7.]))
  >>> `xss_, _ = center(xss_, -xss_means)`
  >>> `int(torch.all(torch.eq(xss, xss_)).item())`
  1

  >>> `xss = torch.arange(12.).view(3,2,2)`
  >>> `xss_, xss_means = center(xss)`
  >>> `xss_means.shape`
  torch.Size([2, 2])
  >>> `xss_, _ = center(xss_, -xss_means)`
  >>> `int(torch.all(torch.eq(xss, xss_)).item())`
  1
  """
  if shift_by is None:
    xss_means = xss.mean(0)
    return xss - xss_means, xss_means
  else:
    return xss - shift_by, xss.mean(0)

def normalize(xss, scale_by = None, **kwargs):
  """Normalize data without dividing by zero.

  See the documentation for the function `center`. This function
  is entirely analogous. The data are assumed to be indexed by
  the first dimension.

  More precisely, let `(d0, d1,`...`, dn)` denote the shape of `xss`.
  In case `scale_by` is not `None`, then the `(i0, i1,` ..., `in)` ent-
  ry of `xss` is divided by the `(i1, i2,`..., `in)` entry of `scale_`
  `by` unless that entry of `scale_by` is (nearly) 0, in which case
  the `(i0, i1,` ...`, in)` entry of `xss` is left unchanged. In oth-
  er words, columns of `xss` whose standard deviation is essent-
  ially zero are left alone; the others are normalized so that
  their standard deviation is 1.0.

  The default, `scale_by=None` is equivalent to setting `scale_by=`
  `xss.std(0)` and leads to the first returned tensor being `xss`
  scaled so that each of its 'columns' have standard deviation 1
  (or were left alone, if that column has essentially no stand-
  ard deviation).

  Args:
    $xss$ (`torch.Tensor`) A tensor whose columns, when thought of
        as being indexed by its first dimension, is to be norm-
        alized.
    $scale_by$ (`torch.Tensor`) A tensor of shape `xss.shape[1:]`.
        Default: `None`.
  Kwargs:
    $unbiased$ (`bool`): If unbiased is `False`, divide by `n` instead
        of `n-1` when computing the standard deviation. Default:
        `True`.
    $threshold$ (`float`): Threshold within which the st. dev. of
        a column is considered too close to zero to divide by.
        Default: `1e-6`.

  Returns:
    `(torch.Tensor, torch.Tensor)`. A tuple of tensors the first
        of which is `xss` normalized with respect to the first
        dimension, except that those columns with standard dev
        less than a small threshold are left unchanged; the se-
        cond is `xss.std(0)`.

  Example:
  >>> `xss = torch.tensor([[1., 2., 3.], [6., 7., 8.]])`
  >>> `xss`
  tensor([[1., 2., 3.],
          [6., 7., 8.]])
  >>> `xss, xss_stdevs = normalize(xss, unbiased = False)`

  The columns of xss are now normalized:
  >>> `xss`
  tensor([[0.4000, 0.8000, 1.2000],
          [2.4000, 2.8000, 3.2000]])

  The stand. devs of the original columns:
  >>> `xss_stdevs`
  tensor([2.5000, 2.5000, 2.5000])

  Let us check that the new columns are normalized:
  >>> `_, xss_stdevs = normalize(xss, unbiased = False)`
  >>> `xss_stdevs`
  tensor([1., 1., 1.])

  More tests and examples:
  >>> `xss = torch.tensor([[1.,2,3], [6,100,-11]])`
  >>> `xss.std(0)`
  tensor([ 3.5355, 69.2965,  9.8995])
  >>> `scale_by = 2.0*torch.ones(xss.shape[1:])`
  >>> `scale_by`
  tensor([2., 2., 2.])
  >>> `xss, stdevs  = normalize(xss, scale_by, unbiased=False)`
  >>> `stdevs`
  tensor([ 1.2500, 24.5000,  3.5000])
  >>> `xss.std(0)/math.sqrt(2)`
  tensor([ 1.2500, 24.5000,  3.5000])

  >>> `xss = torch.tensor([[1., 2, 3], [1, 7, 3]])`
  >>> `xss, stdevs = normalize(xss, unbiased = False)`
  >>> `xss`
  tensor([[1.0...
  >>> `stdevs`
  tensor([0.0...
  """
  # add and assert checking that scale_by is right dim.
  #xss_stdevs = xss.std(0, unbiased)
  #xss_stdevs_no_zeros = xss_stdevs.clone()
  #xss_stdevs_no_zeros[xss_stdevs_no_zeros < 1e-7] = 1.0
  #if isinstance(scale_by, torch.Tensor):
  #  scale_by_no_zeros = scale_by.clone()
  #  scale_by_no_zeros[scale_by_no_zeros < 1e-7] = 1.0
  #  new_xss = xss.div(scale_by_no_zeros)
  #else:
  #  new_xss = xss.div(xss_stdevs_no_zeros)
  #return new_xss, xss_stdevs
  assert isinstance(xss, torch.Tensor),du.utils._markup(
      f'`xss` must have type tensor not {type(xss)}')
  assert isinstance(scale_by, torch.Tensor) or scale_by is None,du.utils._markup(
      f'`scale_by` must be None or have type tensor not {type(scale_by)}')
  du.utils._check_kwargs(kwargs,['unbiased', 'threshold'])
  unbiased = kwargs.get('unbiased',True)
  threshold = kwargs.get('unbiased',1e-6)
  if scale_by is None:
    xss_stdevs = xss.std(0, unbiased)
    xss_stdevs_no_zeros = xss_stdevs.clone()
    xss_stdevs_no_zeros[xss_stdevs_no_zeros < 1e-7] = 1.0
    return xss.div(xss_stdevs_no_zeros), xss_stdevs
  else:
    scale_by_no_zeros = scale_by.clone()
    scale_by_no_zeros[scale_by_no_zeros < 1e-7] = 1.0
    newxss = xss.div(scale_by_no_zeros)
    return xss.div(scale_by_no_zeros), newxss.std(0, unbiased)

def standardize(xss, means=None, stdevs=None, **kwargs):
  """Standardize (a minibatch of) data w/r to `means` and `stdevs`.

  Think of the tensor `xss` as holding examples of data where the
  the first dimension of `xss` indexes the examples. Suppose that
  both tensors `means` and `stdevs` are provided, each of size `xss`
  `.shape[1:]`. Then `standardize` returns a tensor of shape `xss.sh`
  `ape` whose `(i_0, i_2,`...`, i_n)`th entry is

  `(xss_(i_0,`...`,i_n)-means_(i_1,`...`,i_n))/stdevs_(i_1,`...`,i_n).`

  As a simple example, if `xss` is a 100x1 tensor consisting of
  normal data centered at 7 and of width 3 then `standardize` can
  be used to compute the z-scores of elements of `xss` with respect
  to the, in this case, mean and standard deviation of the sin-
  gle dimensional features.

  >>> `xss = 7 + 3 * torch.randn(100).view(100,1)`
  >>> `zss = standardize(xss, xss.mean(0), xss.std(0))`
  >>> `zss.shape`
  torch.Size([100, 1])
  >>> `torch.allclose(zss.mean(0),torch.zeros(1),atol=1e-4)`
  True
  >>> `torch.allclose(zss.std(0),torch.ones(1),atol=1e-4)`
  True

  More generally, below, entries in `xss` are standardized with
  respect to each of the 6 feature dimensions.

  >>> `xss = 50 + torch.randn(1000, 2, 3)`
  >>> `zss = standardize(xss, means = xss.mean(0))`
  >>> `torch.allclose(zss.mean(0), torch.zeros(2, 3), atol=1e-4)`
  True

  In ML, we sometimes wish to standardize testing data with re-
  spect to the means and stdevs of training data. During model
  training, `xss` is often a mini-batch (that we simply standard-
  ize w/r to `xss`'s own means and/or standard deviations.

  For convenience, if one wishes to standardize a single examp-
  le, `xs`, one can simply call `standardize(xs, ...)` rather than
  bothering with say `xs.unsqueeze(0)`:

  >>> `xs = torch.tensor([3., 4, 5])`
  >>> `means = torch.tensor([1., 1, 1])`
  >>> `stdevs = torch.tensor([2., 2, 1])`
  >>> `standardize(xs, means, stdevs)`
  tensor([1.0000, 1.5000, 4.0000])

  Note: division by zero is avoided by replacing any entry in
  `stdevs` that is close to 0.0 with 1.0:

  >>> `xs = torch.tensor([3., 4, 5])`
  >>> `stdevs = torch.tensor([2., 2, 0])`
  >>> `standardize(xs, None, stdevs)`
  tensor([1.5000, 2.0000, 5.0000])

  Args:
    $xss$ (`tensor`): If we denote the size of `xss` by `(d_0, d_1,...`
        `, d_n)`, then we regard `xss` as `d_0` examples of data. For
        a single example, `(1, d_1,` ...`, d_n)` and `(d_1,` ...`, d_n)`
        are treated equivalently.
    $means$ (`tensor`): Tensor of shape `(d_1, d_2,` ...`, d_n)` or `(1,`
        `d_1, d_2,` ...`, d_n)`. Default: `None`, which is equivalent
        to `means=torch.zeros(xss.shape[1:])`.
    $stdevs$ (`tensor`): Same shape restriction as that of `means`.
        Entries within a threshold of 0.0 are effectively repl-
        aced with 1.0 so as not to divide by zero. The default,
        `None` is equivalent to `stdevs=torch.ones(xss.shape[1:])`.

  Kwargs:
    $threshold$ (`float`): Threshold within which the st. dev. of
        a column is considered too close to zero to divide by.
        Default: `1e-6`.

  Returns:
    `torch.tensor`. A tensor of the same shape as `xss`.
  """
  # SHOULD THIS BE IN NORMALIZE??
  #if stdevs is not None:
  #  assert len(xss) > 1 or unbiased == False,\
  #      'len(xss) is 1 but unbiased is: '+str(unbiased)
  #if means is not None:
  #  assert means.shape == xss.shape[1:] or \
  #      means.shape == torch.Size([1]) + xss.shape[1:]
  #if stdevs is not None:
  #  assert stdevs.shape == xss.shape[1:] or \
  #      stdevs.shape == torch.Size([1]) + xss.shape[1:]
  du.utils._check_kwargs(kwargs,['threshold'])
  threshold = kwargs.get('unbiased',1e-6)
  if isinstance(means, torch.Tensor):
    if isinstance(stdevs, torch.Tensor):
      return normalize(center(xss, means)[0], stdevs, threshold=threshold)[0]
    elif stdevs is None:
      return center(xss, means)[0]
  elif isinstance(stdevs, torch.Tensor):
    return normalize(xss, stdevs, threshold=threshold)[0]
  else:
    return xss

def online_means_stdevs(data, *transforms_, batchsize=1):
  """Online compute the means and standard deviations of data.

  Here 'online' means that the examples housed in `data` are tak-
  en `batchsize` at time when computing the means and standard
  deviations. Hence, the `data` need not fit in memory; likewise,
  examples augmented by `transforms_` need not be pre-written to
  the filesystem.

  Args:
    $data$ (`Union[Tuple[tensor], DataLoader]`): Either a tuple of
        tensors each of whose first dimension indexes the exam-
        ples of the data or an instance of `torch.utils.data.`
        `DataLoader` that yields minibatches of such tuples.
    $transforms_$ (`Tuple[torchvision.transforms]`): If you wish to
        compute means and stdevs of augmented data, consider
        defining an appropriate instance of `DataLoader` and add-
        ing your transformation to that; but, you can include
        as many transformations as you wish so long as data is
        a tuple of length 1 or a dataloader yielding those.
    $batchsize$ (`int`): If `data` is a tensor, then an instance of
        `DataLoader` is created with this `batchsize`. If `data` is
        already an instance of `DataLoader`, this argument is ig-
        nored. Default: `1`.

  Returns:
    `Tuple[Tuple[torch.Tensor]]`. A tuple the first tuple of
        which is (means, stdevs) of the first tensor in `data`,
        the second tuple of which is (means, stdevs) of second
        tensor, etc.

  Examples:

  Pass just a tensor:
  >>> data = torch.arange(100.).view(50,2)
  >>> stats, = online_means_stdevs((data,))
  >>> means, stdevs = stats
  >>> means, stdevs
  (tensor([49., 50.]), tensor([28.8617, 28.8617]))

  Add a simple transformation:
  >>> import torchvision.transforms as T
  >>> online_means_stdevs((data,), T.Lambda(lambda xss: xss+100))
  ((tensor([ 99., 100.]), tensor([57.7321, 57.7321])),)

  Now wrap the tensor in a DataLoader instance:
  >>> dataset = torch.utils.data.TensorDataset(data)
  >>> loader = torch.utils.data.DataLoader(dataset, batch_size=25)
  >>> (means, stdevs), = online_means_stdevs(loader)
  >>> means, stdevs
  (tensor([49., 50.]), tensor([28.8617, 28.8617]))

  With a batchsize that doesn't divide the no. of examples:
  >>> loader = torch.utils.data.DataLoader(dataset, batch_size=37)
  >>> online_means_stdevs(loader)
  ((tensor([49., 50.]), tensor([28.8617, 28.8617])),)

  With batchsize=1:
  >>> loader = torch.utils.data.DataLoader(dataset, batch_size=1)
  >>> online_means_stdevs(loader)
  ((tensor([49., 50.]), tensor([28.8617, 28.8617])),)

  With two tensors simultaneously:
  >>> feats = torch.arange(100.).view(50,2)
  >>> targs = torch.arange(50.).view(50,1)
  >>> feats_stats, targs_stats = online_means_stdevs((feats, targs))
  >>> feats_stats
  (tensor([49., 50.]), tensor([28.8617, 28.8617]))
  >>> targs_stats
  (tensor([24.5000]), tensor([14.4309]))

  Using a dataloader
  >>> dataset = torch.utils.data.TensorDataset(feats, targs)
  >>> loader = torch.utils.data.DataLoader(dataset, batch_size=17)
  >>> feats_stats, targs_stats = online_means_stdevs(loader)
  >>> feats_stats
  (tensor([49., 50.]), tensor([28.8617, 28.8617]))
  >>> targs_stats
  (tensor([24.5000]), tensor([14.4309]))
  """
  #Notes:
  #  - This works on one channel (image) data
  #Todo:
  #  - Generalize this to work on multichannel and calls this?
  #    On say 3 channel images, try passing a transform that flattens
  #    those channels to 3 dimensions. Then tack on targets and have...
  #  - Work out a numerically stable version of the online variance algo.
  #    (possibly a batched version of what's on wikipedia).
  #  - The stdevs here are always biased.
  #  - a passed loader batchsize overrides arg batchsize
  #  - adjust so last batch can be smaller ... DONE

  if isinstance(data, tuple) and all(isinstance(x, torch.Tensor) for x in data):
    loader = torch.utils.data.DataLoader(
        dataset = torch.utils.data.TensorDataset(*data),
        batch_size = batchsize,
        num_workers = 0)
  else:
    assert isinstance(data, (torch.utils.data.DataLoader,_DataLoader)),\
        du.utils._markup(
            '`data` must be a tuple of tensors or an instance of either `torch.`'
            '`utils.data.DataLoader` or `du.lib._DataLoader` yielding such mini`'
            '-batches of such tuples.')
    if transforms_!=():
      print(du.utils._markup('$warning$ (from online_means_stdevs):'
          '|best practice is to put transforms in a dataloader|'))
    loader = data
    batchsize = loader.batch_size

  # initialize all the means and variances
  statss = []
  for batch in loader:
    for tensor in batch:
      means = torch.zeros(tensor[0].size()) # BxHxW
      variances = torch.zeros(tensor[0].size()) # BxHxW
      statss.append([means, variances])
    break
  if transforms_ != ():
    assert len(loader.dataset[0]) == 1, du.utils._markup(dedent("""
        If `data` is a tuple of tensors, transforms can only be used when `data`
        is a tuple consisting of a single tensor. Consider breaking up your
        data and calling this on individual pieces or use a dataloader that
        that includes your transforms."""))
    transforms_ = [torchvision.transforms.Lambda(lambda xs: xs)]+list(transforms_)
    # batch update the means and variances
    m = 0
    for transform in transforms_:
      for xss in loader: # loader kicks out tuples; so do
        xss = transform(xss[0])  # <-- this; xs is now a BxHxW tensor
        batchsize = len(xss)
        prev_means = statss[0][0]
        batch_means = xss.mean(0)
        denom = m + batchsize
        statss[0][0] = (m*statss[0][0] + batchsize*batch_means)/denom
        statss[0][1] = m*statss[0][1]/denom + batchsize*xss.var(0,False)/denom \
            + m*batchsize/(denom**2)*(prev_means - batch_means)**2
        m += batchsize
    statss[0][1].pow_(0.5)
  else:
    m = 0
    for minibatch in loader:
      batchsize = len(minibatch[0])
      prev_means = [item[0] for item in statss]
      batch_means = [xss.mean(0) for xss in minibatch]
      denom = m + batchsize
      for stats, xss, prev_mean, batch_mean in \
          zip(statss, minibatch, prev_means, batch_means):
        stats[0] = (m*stats[0] + batchsize*batch_mean)/denom
        stats[1] = m*stats[1]/denom + batchsize*xss.var(0,False)/denom \
            + m*batchsize/(denom**2)*(prev_mean - batch_mean)**2
      m += batchsize
    [stats[1].pow_(0.5) for stats in statss]
  return tuple(tuple(x) for x in statss)

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
          tly that of `model.parameters()` (except that it's a
          list instead of a generator) but with its tensors ini-
          tialized to be all zeros.
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

    An instance of this can be passed to `du.lib.train` via the
    parameter `learn_params`.
    """
    def __init__(self, model, lr = 0.01, mo = 0.9):
        """Constructor.

        Set instance variables `lr` and `mo` and create an instance
        variable `z_params` which is essentially a zeroed out clone
        of `model.parameters()`.

        Args:
          $lr$ (`float`): The learning rate during training.
          $mo$ (`float`): The momentum during training.
        """
        super().__init__(lr)
        self.mo = mo
        self.z_params = copy_parameters(model)

    def __str__(self):
        """Append momentum info to string rep of the base class.

        """
        return super().__str__() + ', momentum: ' + du.utils.format_num(self.mo)

    def set_device(self, device):
        """Send `z_params` to live on device.

        """
        for param in self.z_params:
            param = param.to(device)

    def update(self, params):
        """Update the learning hyper-parameters.

        Update the parameters using momentum.

        Args:
          $parameters$ (`generator`): The parameters (in the form of
              an iterator of tensors) to be updated.
        """
        for z_param, param in zip(self.z_params, params):
            z_param = z_param.mul_(self.mo).add_(param.grad.data)
            param.data.sub_(z_param * self.lr)

def _tuple2dataset(tup):
  """Return instance of Dataset.

  If you don't need any transforms, this a quick way to create
  a dataset from a tuple that fits well in memory.

  >>> feats = torch.rand(40, 2); targs = torch.rand(40, 1)
  >>> dataset = _tuple2dataset((feats, targs))
  >>> len(dataset[0])
  2
  >>> len(dataset[0][0]), len(dataset[0][1])
  (2, 1)
  """
  class Data(torch.utils.data.Dataset):
    def __init__(self, tup):
      self.tup = tup

    def __len__(self):
      return len(self.tup[0])

    def __getitem__(self, idx):
      return tuple([tup[j][idx] for j in range(len(self.tup))])
  return Data(tup)

# keeping this since it's faster than DataLoader
class _DataLoader:
  """Emulate `torch.utils.data.DataLoader`.

  An instance of this can be used in the same way that one uses
  an instance of `DataLoader`. If you are not using transforms
  and the totality of your data fits in RAM, this can be faster
  that `DataLoader`.
  """

  def __init__(self, data_tuple, batch_size, shuffle=False):
    """
    Args:
      $data_tuple$ (Tuple[`torch.Tensor`]).
      $batch_size$ (`Int`).
      $shuffle$ (`bool`). Default: `False`.

    Examples:
    >>> `xss = torch.rand(144).view(12,3,4)`
    >>> `yss = torch.arange(12).view(12,1)`
    >>> `dl =_DataLoader(data_tuple=(xss,yss),batch_size=6)`
    >>> `for mb in dl:`
    ...     `print(mb[0].size(), mb[1].size())`
    torch.Size([6, 3, 4]) torch.Size([6, 1])
    torch.Size([6, 3, 4]) torch.Size([6, 1])
    >>> `len(dl)`
    2

    >>> `dl =_DataLoader(data_tuple=(xss,yss),batch_size=12)`
    >>> `for mb in dl:`
    ...     `print(mb[0].size(), mb[1].size())`
    torch.Size([12, 3, 4]) torch.Size([12, 1])

    Note that this, like DataLoader, produces smaller last
    minibatches if the batch_size does not divide the length.

    >>> `dl =_DataLoader(data_tuple=(xss,yss),batch_size=7)`
    >>> `for mb in dl:`
    ...     `print(mb[0].size(), mb[1].size())`
    torch.Size([7, 3, 4]) torch.Size([7, 1])
    torch.Size([5, 3, 4]) torch.Size([5, 1])
    >>> `len(dl)`
    2

    >>> `dataset = torch.utils.data.TensorDataset(xss,yss)`
    >>> `dl = torch.utils.data.DataLoader(dataset, batch_size=7)`
    >>> `for mb in dl:`
    ...     `print(mb[0].size(), mb[1].size())`
    torch.Size([7, 3, 4]) torch.Size([7, 1])
    torch.Size([5, 3, 4]) torch.Size([5, 1])
    """
    self.len = len(data_tuple[0])
    # we don't really use self.dataset as an instance of Dataset, here. But we
    # can now # get len(self.dataset) as with instances of DataLoader. So self.
    # dataset is just storing the tuple, really.
    self.dataset = _tuple2dataset(data_tuple)
    self.batch_size = batch_size
    self.shuffle = shuffle

  def __iter__(self):
    if self.shuffle:
      self.indices = torch.randperm(self.len)
    else:
      self.indices = torch.arange(self.len)
    self.idx = 0
    return self

  def __len__(self):
    return math.ceil(len(self.dataset)/self.batch_size)

  def __next__(self):
    if self.idx > self.len - 1:
      raise StopIteration
    #minibatch = tuple([t.index_select(0,
    #    self.indices[self.idx: self.idx + self.batch_size]) for\
    #        t in self.dataset])
    minibatch = tuple([t.index_select(0,
        self.indices[self.idx: self.idx + self.batch_size]) for\
            t in self.dataset.tup])
    self.idx += self.batch_size
    return minibatch

def _evaluate(model, dataloader, crit, device):
  """Return ave. value of a metric on a model and a dataloader.

  This uses the metric `crit` (which assumed to return a number
  that is an average per example) to evaluate `model` on data
  wrapped in `dataloader`; on 'device' it accumulates and then ret-
  urns the loss. `dataloader` can be an instance of `torch.utils.`
  `data.DataLoader` or `du.lib._DataLoader`.

  Args:
    $model$ (`nn.Module`): The model to applied.
    $dataloader$ (`Union[DataLoader, _DataLoader]`)
    $crit$ (`function`): A function that maps a mini-batch output
        by `dataloader` to a float representing an average (per
        example) value.
    $device$ (`Tuple[int,torch.device]`): The model should already
        be on this device. The mini-batches are moved to this
        device just before passing them through the model and
        evaluating with criterion. As usual: if this a non-neg-
        ative `int`, then it use that GPU; -1, use the last GPU;
        -2 force use of the CPU.

  Returns:
    `float`. The average loss over one epoch of `dataloader`.
  """
  device = du.utils.get_device(device) if isinstance(device,int) else device
  accum_loss = 0.0
  #num_examples = 0
  for minibatch in dataloader:
    accum_loss += crit(model(
        *map(lambda x: x.to(device), minibatch[:-1])), minibatch[-1].to(device))
    #num_examples += len(minibatch[0])
  return accum_loss/len(dataloader)

#def _batch2r2(yhats, yss, device=-2):
#  assert len(yhats) == len(yss)
#  return torch.square(yhats-yss)

#def _mse(model, dataloader, device=-2):
#  return _evaluate(model, dataloader, crit=nn.functional.mse_loss, device=device)

#def _rmse(model, dataloader, device=-2):
#  return _evaluate(
#      model,
#      dataloader,
#      crit= lambda xss, yss: torch.sqrt(nn.functional.mse_loss(xss, yss)),
#      device=device)

_rmse = lambda xss, yss: torch.sqrt(nn.functional.mse_loss(xss, yss))

# can be removed now
#def _class_accuracy(model, dataloader, device=-2):
#  """Return proportion correct.
#
#  """
#  return _evaluate(model, dataloader, crit=_batch2class_accuracy, device=device)

def train(model, crit, train_data, **kwargs):
  """Train a model.

  The loss printed to the console is the average loss per samp-
  le over an epoch as that average is accumulated during train-
  ing. If the number of training examples is divisible by the
  batchsize then, during training, the model sees each example
  in the training data exactly once during an epoch.

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
  ally, one can accept the default (`gpu`=`(-1,)`) which sends all
  computations to the (last of any) found GPU(s) and, if there
  are no GPU(s), to the (first) CPU (thread).

  Just before mini-batches are forward-passed through the model
  during training, they are moved from the CPU to the training
  device determined by the first entry in the tuple `gpu`. Mean-
  while, the model has always been moved to the training device
  at the beginning of training.

  !Note on validation and efficiency!

  In order to provide an option that trains as efficiently as
  possible, unless `graph` is positive, any validation data that
  may have been passed as an argument of `valid_data` is ignored;
  that is, the model is simply trained on the provided training
  data, and the loss per epoch is displayed to the console. Use
  the default `gpu = (-1,)` to train on the fastest available de-
  vice.

  You can set `graph` to be positive (and forego validation) in
  order to real-time graph the losses per epoch at cost in time
  but at no cost in VRAM (assuming you have GPU(s)) if you set
  `gpu = (-1, -2)`. Here the -1 leads to training on the GPU and
  the -2 causes validation during training to take place on the
  CPU.  Moreover, the training data, for the purpose of valida-
  remains on the CPU, thus freeing VRAM for training (at the
  expense of time efficiency since validation is likely slower
  on a CPU).

  By default, any provided `valid_data` resides on the device on
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
    $train_data$ (`Union[Tuple[torch.Tensor],DataLoader]`) Either a
        tuple consisting of 2 or 3 tensors (as described below)
        or an instance of `torch.utils.data.DataLoader` yielding
        such tuples.
        Passing a length 3 tensor is only necessary when train-
        ing a recurrent net on variable length inputs. In that
        case, the triple of tensors must be of the form
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
    $valid_data$ (`Union[Tuple[torch.Tensor],DataLoader]`):
        (Optional) data on which to validate the model in the
        form of a tuple of length 2 or 3 (that is, matching the
        length of `train_data`) or an instance of `torch.data.util`
        `s.data.DataLoader` yielding such tensors. See also the
        documentation below for `valid_metric`.
        The loss on validation data is computed each epoch; but
        `valid_data` is not shown to the model as part of back-
        propagation. Default: `None`.
    $valid_metric$ (`Union[bool, function]`): If this is not `False`
        and `graph` is positive then, using a metric, `model` is
        validated, during training, on any data that is passed
        via `valid_data` with the results displayed in real time
        on a graph. The model is also validated on `train_data`;
        and those results are also diplayed. If `valid_data` is
        `None`, then `model` is only validated on training data
        and only those results are diplayed.
        The metric used for validation can be specified by pro-
        viding it as an argument here; however, if the argument
        here is simply `True`, then the metric used automatically
        becomes explained variance in the case that the targets
        of `train_data` are floats (as in, for example, a regres-
        sion problem) or proportion of correct predictions if
        those targets are integers (as in, e.g., a classifica-
        tion problem).
        Alternatively, `valid_metric` can be a function that maps
        tuples of the form `(model(xss), yss)` to floats; though,
        if `valid_data` as a dataloader, then the function should
        output an average per example on batches.
        For a regression problem, one could put
          `valid_metric=torch.nn.functional.mse_loss`, or
          `valid_metric=du.lib._rmse_loss`.
        The last option is equivalent to
          `valid_metric=lambda xss, yss: torch.sqrt(`
               `torch.nn.functional.mse_loss(xss, yss))`.
        Note that:
        - ~expected variation~ (the automatic metric for a reg-
          ression problem) may not be the best choice though,
          for an OLS model it is ~r-squared~, the coefficient of
          determination.
        - to simply train the model as efficiently as possible,
          set `graph = 0` which disables all validation;
        - or, set `valid_metric=False`, to disable all validation
          and just graph (if, e.g., `graph = 1`) the loss.
        Default: `True`.
    $learn_params$
        (`Union[dict,LearnParam_, torch.optim.Optimizer]`): The
        training, or 'learning', hyperparameters in the form of
        an instance of `LearnParams_`; or, for basic functionali-
        ty, a `dict` that maps the string 'lr', and optionally
        'mo', to `float`s; or an instance of `torch.optim.Optimizer`
        Default: `{'lr': 0.1}`.
    $bs$ (`int`): The mini-batch size where -1 forces batch gradi-
        ent descent (i.e. feed-forwarding all training examples
        before each back-propagation). Default: `-1`.
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
    $verb$ (`int`): Verbosity; 0, silent; 1, just timing info; 2,
        also print device notes; 3, add loss per epoch. Def.:`3`.
    $gpu$ (`Tuple[int]`): Tuple of `int`s of length 1 or 2 where the
        first entry determines the device to which the model is
        moved and, in fact, on which the forwarding and back-
        propagation through the model takes place during train-
        ing.
        The second entry determines the device to which the mo-
        del is deep copied (if necessary) for the purpose of
        validation including validation agaisnt any test data
        provided. If this is a length 1 tuple, then that number
        is used to determine both devices.
        If no GPUs are present, then accept the default. Other-
        wise an `int` determines the GPU to use for training/val-
        idating. When GPU(s) are present, set an entry of the
        tuple to an `int` to select the corresponding GPU, or
        to -1 to use the last GPU found (and to use the CPU if
        no GPU is found), or to -2 to override using a found
        GPU and instead use the CPU. Default: `(-1, -1)`.
    $args$ (`argparse.Namespace`): With the exception of `valid_data`
        `valid_metric`, and this argument, all `kwargs` can be
        passed to `train` via attributes (of the same name) of an
        instance of `argparse.Namespace`. Default: None.

        Note: arguments that are passed explicitly via their
        parameter above |override| any of those values passed via
        `args`.

  Returns:
    `nn.Module`. The trained model (still on the device determin-
        ed by `gpu`).
  """
  # this is train
  # check and process kwargs
  du.utils._check_kwargs(kwargs,
      ['valid_data', 'learn_params', 'bs', 'epochs', 'graph',
       'print_lines', 'verb', 'gpu', 'valid_metric', 'args'])
  valid_data = kwargs.get('valid_data', None)
  args = kwargs.get('args', None)
  if args == None:
    class args: pass # a little finesse if args wasn't passed
  else:
    for kwarg in ['learn_params', 'bs', 'epochs', 'graph',
        'print_lines', 'verb', 'gpu']:
      if kwarg in kwargs and kwarg in vars(args).keys():
        print(du.utils._markup('$warning$ (from train):'
            f'|argument passed via parameter| `{kwarg}`'
            f' |overriding| `args.{kwarg}`'))
  bs = kwargs.get('bs', -1 if not hasattr(args,'bs') else args.bs)
  verb = kwargs.get('verb', 3 if not hasattr(args,'verb') else args.verb)
  gpu = kwargs.get('gpu', (-1,) if not hasattr(args,'gpu') else args.gpu)
  epochs=kwargs.get('epochs', 10 if not hasattr(args,'epochs') else args.epochs)
  valid_metric = kwargs.get('valid_metric', True)
  learn_params = kwargs.get( 'learn_params',
      {'lr': 0.1 if not hasattr(args,'lr') else args.lr,
          'mo': 0.0 if not hasattr(args,'mo') else args.mo} if \
          not hasattr(args,'learn_params') else args.learn_params)
  print_lines = kwargs.get( 'print_lines',
      (7, 8) if not hasattr(args,'print_lines') else args.print_lines)
  if len(print_lines) > 1: print_init, print_last = print_lines
  else: print_init, print_last = print_lines[0], print_lines[0]
  graph = kwargs.get('graph', 0 if not hasattr(args,'graph') else args.graph)
  graph = 1 if graph is True else graph
  assert isinstance(graph, int) and graph >= 0,\
      du.utils._markup(f'`graph` must be a non-negative integer, not {graph}.')

  start = time.time() # start (naive) timing here

  # get devices determined by the gpu argument
  if isinstance(gpu, (tuple,list)) and len(gpu) == 1:
    gpu = (gpu[0], gpu[0])
  else:
    assert isinstance(gpu, (tuple,list)) and len(gpu) > 1
  # The training happens on the model device; training mini-batches are moved
  # just before being forwarded through the model.
  model_dev = du.utils.get_device(gpu[0])
  valid_dev = du.utils.get_device(gpu[1])  # this is where validation happens
  data_dev = torch.device('cpu',0) # this is where the data lives
  if verb > 1:
    print(f'training on {model_dev} (data is on {data_dev})',end='')
    if valid_metric and graph > 0: print(f'; validating on {valid_dev}')
    else: print()

  # is this what you want and where you want it
  if model_dev.type == 'cuda': torch.backends.cudnn.benchmark = True

  # parse the training data and leave it in data_dev memory
  if isinstance(train_data, torch.utils.data.DataLoader):
    has_lengths = True if len(train_data.dataset[0]) > 2 else False
    num_examples = len(train_data.dataset)
  else: # is tuple of tensors; wrap it in an instance of _DataLoader
    assert 2 <= len(train_data) <= 3
    assert all([isinstance(x, torch.Tensor) for x in train_data])
    has_lengths = True if len(train_data) > 2  else False
    num_examples = len(train_data[0])
    if bs <= 0: bs = num_examples
    #  train_data = torch.utils.data.DataLoader(_tuple2dataset(train_data),
    #      batch_size = bs, num_workers=2, shuffle=True, pin_memory = True)
    # Note: using _DataLoader here is faster than using DataLoader
    train_data = _DataLoader(train_data, bs, shuffle=True)

  model = model.to(model_dev) # move the model to the right device
  #if verb > 2: print(model)

  # process learn_params
  has_optim = False
  if isinstance(learn_params, Dict):
    for key in learn_params.keys(): assert key in ['lr','mo'],\
        f"keys of learn_params dict should be 'lr' or 'mo', not {key}."
    assert 'lr' in learn_params.keys(), "input dict must map 'lr' to  a float"
    lr = learn_params['lr']
    #if verb > 1: print('learning rate:', du.utils.format_num(lr), end=', ')
    if 'mo' not in learn_params.keys():
      learn_params = LearnParams_(lr = lr)
      mo = None
    else:
      mo = learn_params['mo']
      #if verb > 1: print('momentum:', du.utils.format_num(mo), end=', ')
      learn_params = Momentum(model, lr = lr, mo = mo)
      learn_params.set_device(model_dev)
    #if verb > 1: print('batchsize:', bs)
  elif isinstance(learn_params, torch.optim.Optimizer):
    has_optim = True
  else:
    assert isinstance(learn_params, LearnParams_), dedent(f"""\
        learn_params must be a dict or an instance of a subclass of
        LearnParams_, not a {type(learn_params)}.""")
    learn_params.set_device(model_dev) # set the device for learn params
    #if verb > 1: print(learn_params, end=', ')
    #if verb > 1: print('batchsize:', bs)

  #else:
  #  #assert isinstance(valid_metric, FunctionType)
  #  valid_metric = _evaluate(model, dataloader, crit=valid_metric , device=device)

  if graph:
    import matplotlib.pyplot as plt # Don't import these until now in case
    import numpy as np              # someone no haz matplotlib or numpy.
    plt.ion()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch', size='larger')
    ax1.set_ylabel('average loss',size='larger')
    ax2 = ax1.twinx()
    ax2.set_ylabel('validation',size='larger');
    xlim_start = 1

    # parse valid_metric and setup v_dation_train
    if isinstance(valid_metric, bool):
      if valid_metric:
        # Setup valid_metric according to whether this looks like a regression
        # or a classification problem.
        for minibatch in train_data:
          if isinstance(minibatch[-1][0], FloatTensor):
            valid_metric = 'regression'
            #valid_metric = nn.functional.l1_loss
            v_dation_train = lambda model:  1-len(train_data)*_evaluate(
                model,
                dataloader=train_data,
                crit=_explained_var(train_data,device=valid_dev),
                device=valid_dev)
          elif isinstance(minibatch[-1][0], IntTensor):
            #valid_metric = _batch2class_accuracy
            v_dation_train = lambda model:  _evaluate(
                model, dataloader=train_data,
                crit=_batch2class_accuracy, device=valid_dev)
          else:
            raise RuntimeError('please specify a function to use for validation')
          break
    elif isinstance(valid_metric, FunctionType):
      # this maps: model -> float
      v_dation_train = lambda model: _evaluate(model, dataloader=train_data,
          crit=valid_metric, device=valid_dev)
    else:
      raise RuntimeError('valid_metric must be boolean or a function')

    # these will hold the losses and validations for train data
    losses = []
    if valid_metric: v_dations = []

    if valid_data and valid_metric:
      # parse the valid_data
      if isinstance(valid_data, torch.utils.data.DataLoader):
        if len(valid_data.dataset) == 0: valid_data = None
        assert len(valid_data.dataset[0]) == len(train_data.dataset[0])
      else:
        if len(valid_data[0]) == 0: valid_data = None
        assert len(valid_data) == 3 if has_lengths else 2
        assert all([isinstance(x, torch.Tensor) for x in valid_data])
        #just use the same batchsize as with training data
        valid_data = _DataLoader(valid_data, bs, shuffle = False)
      # set up v_dation_valid
      if isinstance(valid_metric, FunctionType):
        v_dation_valid=functools.partial(  # this maps:  model -> float
            _evaluate, dataloader=valid_data, crit=valid_metric, device=valid_dev)
      else:  # then valid_metric is output of _explained_var
        if valid_metric == 'regression':
          v_dation_valid = lambda model: 1-len(valid_data)*_evaluate(
              model,
              dataloader=valid_data,
              crit=_explained_var(valid_data, device=valid_dev),
              device=valid_dev)
        else:
          v_dation_valid = lambda model: _evaluate(
              model,dataloader=valid_data,
              crit=_batch2class_accuracy, device=valid_dev)
      # set up loss_valid
      # this also maps:  model -> float
      loss_valid = lambda model: _evaluate(model, dataloader=valid_data,
          crit=crit, device=valid_dev)

      losses_valid=[] # this will hold the losses for test data
      v_dations_valid = [] # this will hold the validations for test data

  # set up console printing
  if print_init == -1 or print_last == -1: print_init, print_last = epochs, -1

  # try to catch crtl-C
  du.utils._catch_sigint()

  # training loop
  for epoch in range(epochs):
    model.train()
    accum_loss = 0
    for minibatch in train_data:
      #print(minibatch[0].size(), minibatch[-1].size()); quit()
      loss = crit(model(
          *map(lambda x: x.to(model_dev), minibatch[:-1])),
          minibatch[-1].to(model_dev))
      accum_loss += loss.item()
      if has_optim: learn_params.zero_grad()
      else: model.zero_grad()
      loss.backward()
      if has_optim: learn_params.step()
      else: learn_params.update(model.parameters())

    # print to terminal
    if print_init * print_last != 0 and verb > 2:
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
      # set some facecolors:
      #blue_fc = 'tab:blue'; red_fc = 'tab:red'  # primary
      #blue_fc = '#799FCB'; red_fc = '#F9665E'    # pastel
      #blue_fc = '#95B4CC'; red_fc = '#FEC9C9'    # more pastel
      #blue_fc = '#AFC7D0'; red_fc = '#EEF1E6'    # more
      #blue_fc = '#c4cfcf'; red_fc = '#cfc4c4'
      blue_fc = '#c4cfcf'; red_fc = '#cfc4c4'

      model.eval()

      with torch.no_grad():  # check that this is what you want
        losses.append(accum_loss*bs/num_examples)

        # copy the model to the valid_dev, if necessary
        model_copy = model if valid_dev == model_dev else\
            copy.deepcopy(model).to(valid_dev)
        model_copy.eval() # and check that this is what you want

        if valid_metric:
          v_dations.append(v_dation_train(model_copy))
        # validate on valid_data
        if valid_data is not None:
          #if has_lengths:   # remove this if-else using  *map stuff as above?
          #  loss=crit(model_copy(test_feats,test_feats_lengths),test_targs).item()
          #else:
          #  loss = crit(model_copy(test_feats), test_targs).item()
          losses_valid.append(loss_valid(model_copy))
          v_dations_valid.append(v_dation_valid(model_copy))

        # (re)draw the actual graphs
        if epoch > epochs - graph:
          xlim_start += 1
        ax1.clear()
        ax2.clear()
        ax1.set_xlabel('epoch', size='larger')
        if valid_data:
          ax1.set_ylabel('average loss (stippled)',size='larger')
        else:
          ax1.set_ylabel('average loss',size='larger')
        ax2.set_ylabel('validation',size='larger')
        xlim = range(xlim_start,len(losses)+1)
        loss_ys = np.array(losses[xlim_start-1:], dtype=float)
        if valid_metric:
          v_dation_ys = np.array(v_dations[xlim_start-1:], dtype=float)
        if valid_data:
          losstest_ys = np.array(losses_valid[xlim_start-1:], dtype=float)
          v_dationtest_ys = np.array(v_dations_valid[xlim_start-1:], dtype=float)
          ax1.plot(xlim,losstest_ys,xlim,loss_ys,color='black',lw=.5)
          ax1.fill_between(xlim,losstest_ys,loss_ys,where = losstest_ys >=loss_ys,
              facecolor=red_fc,interpolate=True, alpha=.8, hatch=5*'.')
          ax1.fill_between(xlim,losstest_ys,loss_ys,where = losstest_ys <=loss_ys,
              facecolor=blue_fc,interpolate=True, alpha=.8,hatch=5*'.')
          ax2.plot(xlim,v_dationtest_ys,xlim,v_dation_ys,color='black',lw=.5)
          ax2.fill_between(xlim,v_dationtest_ys,v_dation_ys,
              where = v_dationtest_ys >=v_dation_ys, facecolor=red_fc,
              interpolate=True, alpha=.8,label='test > train')
          ax2.fill_between(xlim,v_dationtest_ys,v_dation_ys,
              where = v_dationtest_ys <=v_dation_ys,
              facecolor=blue_fc,interpolate=True, alpha=.8,label='train > test')
          ax2.legend(fancybox=True, loc=2, framealpha=0.8, prop={'size': 9})
        else:
          ax1.plot(xlim,loss_ys,color='black',lw=1.2,label='loss')
          ax1.legend(fancybox=True, loc=8, framealpha=0.8, prop={'size': 9})
          if valid_metric:
            ax2.plot(xlim,v_dation_ys,color=blue_fc,lw=1.2,label='validation')
            ax2.legend(fancybox=True, loc=9, framealpha=0.8, prop={'size': 9})
        len_valid_data = len(valid_data.dataset) if valid_data is not None else 0
        plt.title('training on {} ({:.1f}%) of {} examples'.format( num_examples,
            100*(num_examples/(num_examples+len_valid_data)),
            num_examples+len_valid_data))
        try:
          fig.canvas.draw()
          fig.canvas.flush_events()
        except tkinter.TclError:
          plt.ioff()
          exit()
        fig.tight_layout()

  end = time.time()
  if verb > 0:
    if end - start < 60:
      print(f'trained in {(end-start):.1f} sec')
    if end - start > 60:
      print(f'trained in {int((end-start)//60)} min {(end-start)%60:.1f} sec')

  if graph:
    plt.ioff()
    plt.title('trained on {} ({:.1f}%) of {} examples'.format(num_examples,
        100*(num_examples/(num_examples+len_valid_data)),
        num_examples+len_valid_data))
    fig.tight_layout()
    #plt.show(block = True)
    plt.show()

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
    $valid_metric$ (`nn.Module`): The validation metric to use when
        gauging the accuracy of the model on test data. If this is
        `None`, then `crit`, the training criterion, is used.
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
  du.utils._check_kwargs(kwargs,['k','valid_metric','cent_norm_feats',\
      'cent_norm_targs','learn_params','bs','epochs','gpu','verb'])
  du.utils._catch_sigint()
  valid_metric = kwargs.get('valid_metric', None)
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

  if not valid_metric: valid_metric = crit

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
        train_data = (xss_train, yss_train),
        learn_params = learn_params,
        bs=bs,
        epochs=epochs,
        verb=verb-1,
        gpu=(gpu,))

    if cent_feats: xss_test.sub_(xss_train_means)
    if norm_feats: xss_test.div_(xss_train_stdevs)
    if cent_targs: yss_test.sub_(yss_train_means)
    if norm_targs: yss_test.div_(yss_train_stdevs)

    valids[idx//chunklength] = valid_metric(model(xss_test), yss_test)

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
    $valid_metric$ (`nn.Module`): The validation criterion to use
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
  du.utils._check_kwargs(kwargs,['k','bail_after','valid_metric',\
      'cent_norm_feats','cent_norm_targs','learn_params','bs',\
      'epochs','verb','gpu'])
  valid_metric = kwargs.get('valid_metric', None)
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

  if not valid_metric: valid_metric = crit

  while no_improvement < bail_after:

    model, valids = cross_validate_train(
        model = model,
        crit = crit,
        train_data = train_data,
        k = k,
        valid_metric = valid_metric,
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

def _batch2class_accuracy(probdists, yss):
  """Return the proportion correctly classified.

  Args:
    $prob_dists$ (`torch.Tensor`): A tensor of dimension 2 holding,
        for each example, the probability distribution predict-
        ing the correct class. The first dimension must index
        the examples. This argument is, then, the predictions,
        in the form of probability distributions, made by a mo-
        del when fed the features of some set of examples. This
        should often be just `model(xss)`, for example.
    $yss$ (`IntTensor`): A 1-dimensional tensor holding the cor-
        rect class (as some flavor of an `int`) for each example.

  Returns:
    `float`. The proportion of examples correctly predicted.

  Todo:
    - See if moving moving probdists and yss back to CPU
      improves speed (likely not).
  """
  assert len(probdists) == len(yss), dedent(f"""\
      Lengths must be equal, but len(probdists)={len(probdists)}
      and len(yss)={len(yss)}.""")
  assert isinstance(yss, IntTensor), dedent(f"""\
      Argument yss must be a Long-, Int-, or ShortTensor, not {type(yss)}.""")
  accum = 0
  for probdist, ys in zip(probdists, yss):
    if torch.argmax(probdist).item() == ys:
      accum += 1
  return accum/len(probdists)

def class_accuracy(model, data, **kwargs):
  """Return the classification accuracy.

  By default, this returns the proportion correct when using
  `model` to classify the features in `data` using the targets in
  `data` as ground truth; optionally, the confusion 'matrix' is
  displayed as a table - where the columns correspond to ground
  truch and the rows are the class predicted by `model`.

  Args:
    $model$ (`nn.Module`): The trained model.
    $data$ (`Union[Tuple[Tensor], DataLoader`): Either a tuple of
        tensors `(xss, yss)` where `xss` holds the features of the
        data on which to access accuracy (and whose first dim-
        ension indexes the examples to be tested) and `yss` is a
        tensor of dimension 1 holding the corresponding correct
        classes (as `int`s), or an instance of `torch.utils.data`
        `.DataLoader` which yields mini-batches of such 2-tuples.

  Kwargs:
    $classes$ (`IntTensor`): A 1-dimensional tensor holding the nu-
        merical classes. This is naturally `torch.arange(10)` for
        digit classification, for instance. The default is `None`
        which leads to `classes=torch.arange(num_classes)` where
        `num_classes` is the length of the output of `model` on the
        features of a single example.
    $class2name$ (`Dict[int, str]`): A dictionary mapping each num-
        erical class to its classname. (The classnames are only
        used when displaying the confusion matrix.) Def.: `None`.
    $gpu$ (`Union[torch.device, int]`): The GPU to use if there are
        any available. Set this to -1 to use the last GPU found
        or to, if no GPU is found, use the (first) CPU; set to
        -2 to override using any found GPU and instead use the
        CPU. Alternatively, one can set this to an instance of
        `torch.device`. Default: `-1`.
    $show_cm$ (`bool`): If `True`, then display an ascii confusion
        matrix. Default: `False`.
    $color$ (`bool`): Whether to colorize the confusion matrix.
        Default: `True`.

  Returns:
    `float`. The proportion of correct predictions.
  """
  #this is class_accuracy
  #check and get kwargs
  du.utils._check_kwargs(kwargs,['classes','show_cm','class2name','gpu','color'])
  classes = kwargs.get('classes', None)
  show = kwargs.get('show_cm', False)
  class2name = kwargs.get('class2name', None)
  gpu = kwargs.get('gpu', -1)
  color = kwargs.get('color',True)
  device = gpu if isinstance(gpu,torch.device) else du.utils.get_device(gpu)

  model.eval()

  with torch.no_grad():
    #check whether the device already lives on the device determined above
    already_on = list(model.parameters())[0].device
    if (str(device)[:3] !=  str(already_on)[:3] or str(device)[:3] != 'cpu')\
       and device != already_on:
      print(du.utils._markup('$warning$ (from class_accuracy):'), end=' ')
      print(du.utils._markup(f'|model moved from| `{already_on}` to `{device}`'))
      model = model.to(device)

    # Check basic things and set stuff up including creating an appropriate,
    # according to whether the user passed, as argument to probdists, some
    # outputs of a model or a (model, outputs) tuple. And move to device.
    assert isinstance(model, nn.Module)
    if not isinstance(data, torch.utils.data.DataLoader):
      # if user did not pass a DataLaoder then check stuff & wrap in _DataLoader
      assert len(data[0]) == len(data[1]), dedent(f"""\
          The number of features ({len(data[0])}) must be equal to the
          number of targets ({len(data[1])}).""")
      assert (isinstance(data, tuple) and len(data)==2),\
          'If argument data is a tuple, it must have length 2 not {}'.format(
              len(data))
      loader = _DataLoader(data, batch_size=10)
      num_classes =\
          len(model(loader.dataset[0][0].to(device).unsqueeze(0)).squeeze(0))
    else:  # the user passed a DataLaoder
      loader = data
      assert isinstance(loader.dataset[0], tuple) and len(loader.dataset[0])==2,\
          'dataloader should yield 2-tuples'
      num_classes =\
          len(model(loader.dataset[0][0].to(device).unsqueeze(0)).squeeze(0))
    if classes is None:
      classes = torch.arange(num_classes)
    else:
      assert classes.dim() == 1,\
          'The classes argument should be a 1-dim tensor not a {}-dim one.'\
              .format(classes.dim())

    if not show: # just compute the accuracy
      accuracy = _evaluate(model, loader, crit=_batch2class_accuracy, device=device)
    else:
      # compute the entries in the confusion matrix
      cm_counts = torch.zeros(len(classes), len(classes))
      counts = torch.zeros(len(classes))
      num_examples = 0
      for batch in loader:
        for prob, ys in zip(model(batch[0].to(device)), batch[1].to(device)):
          cm_counts[torch.argmax(prob).item(), ys] += 1
        counts += torch.bincount(batch[1], minlength=len(classes))
        num_examples += len(batch[0])
      cm_pcts = cm_counts/num_examples

      # display the confusion matrix
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
              string = du.utils._markup('~'+string+'~', strip = not color)
            print(' '*(cell_length-length)+string, end='')
        n_examples = cm_counts[:,i].sum()
        pct = 100*(cm_counts[i,i]/n_examples) if n_examples != 0 else 0
        if class2name:
          print('  {} ({:.1f}% of {})'.format(class2name[i],pct,int(counts[i])))
        else:
          print(' ({:.1f}% of {})'.format(pct, int(counts[i])))
      accuracy = torch.trace(cm_pcts).item()
  return accuracy

# change this to explained_var and change throughout
def _explained_var(loader, device, mean_zero = False):
  """Helper to compute the explained variation (variance).

  Under certain conditions (e.g., poly lin regression), one has
  the ANOVA decomposition
                       TSS = RSS + ESS
  where
  TSS = (yss - yss.mean(0)).pow(2).sum()  #total sum of squares
  RSS = (yss - yhats).pow(2).sum()    # residual sum of squares
  ESS=(yhats-yss.mean(0)).pow(2).sum()#explained sum of squares.

  So, under preferred conditions, one computes the explained
  variance, resp. unexplained variance as a proportion of tot-
  al  variance: ESS/TSS or RSS/TSS.

  However, absent preferential conditions one can use 1-RSS/TSS
  for explained variance.

  This is a helper function for computing explained variance.
  Both `train_loader` and `test_loader` are assumed to be in-
  stances of Dataloader that yield feats and targets as tuples
  of tensors.

  Args:
    $loader$ (`Dataloader`)
    $device$
    $mean_zero$

  Returns:
    `function`. Function that maps a pair of tensors to a float.

  >>> `yhatss = torch.arange(4.).unsqueeze(1)`
  >>> `yss = torch.tensor([-1., 5., 2., 3.]).unsqueeze(1)`
  >>> loader = _DataLoader((yhatss, yss), batch_size=2)
  >>> `1 - _explained_var(loader, 'cpu')(yhatss, yss).item()`
  0.09333...
  """
  if mean_zero:
    yss_mean = torch.tensor(0.)
  else:
    yss_mean = online_means_stdevs(loader)[1][0]
  TSS = 0
  for minibatch in loader:
    TSS += (minibatch[1]-yss_mean).pow(2).sum().item()
  return lambda yhats, yss: (yhats-yss).pow(2).sum().item()/TSS

def explained_var(model, data, **kwargs):
  """Compute the explained variance.

  Returns the coefficient of determination of two 2-d tensors
  (where the first dimension in each indexes the examples), one
  holding the `yhatss` (the predicted outputs) and the other hol-
  ding the true outputs, `yss`.

  Args:
    $model$ (`nn.Module`): The trained model.
    $data$ (`Union(Tuple[Tensor], DataLoader)`): Either a tuple of
        tensors `(xss, yss)` where `xss` are the features of the
        data and `yss` (assumed to be of shape len(yss) by 1) are
        the targets or an instance of `torch.utils.data.DataLoad`
        `er` that yields such tuples.

  Kwargs:
    $return_error$ (`bool`): If `False`, return the proportion of the
        variation explained by the regression line. If `True`,
        return 1 minus that proportion. Default: `False`.
    $gpu$ (`Union[torch.device, int]`): The GPU to use if there are
        any available. Set this to -1 to use the last GPU found
        or to, if no GPU is found, use the (first) CPU; set to
        -2 to override using any found GPU and instead use the
        CPU. Alternatively, one can set this to an instance of
        `torch.device`. Default: `-1`.

  Returns:
    `float`. The proportion of variation explained by the model
        (as compared to a constant model) or (optionally) 1 mi-
        nus that proportion (i.e., the proportion unexplained).
  """
  # this is explained_var
  du.utils._check_kwargs(kwargs,['return_error','gpu'])
  return_error = kwargs.get('return_error', False)
  gpu = kwargs.get('gpu', -1)
  device = gpu if isinstance(gpu, torch.device) else du.utils.get_device(gpu)

  if isinstance(data, tuple):
      assert len(data) == 2 and len(data[0]) == len(data[1])
      data = _DataLoader(data, batch_size = len(data[0]))
  else:
      assert isinstance(data,(torch.utils.data.DataLoader,_DataLoader))
  error = len(data)*_evaluate(
      model, dataloader=data, crit=_explained_var(data,device=gpu), device=device)
  return error if return_error else 1-error

#  if not isinstance(yhatss, torch.Tensor):
#    assert isinstance(yhatss, (tuple,list)),\
#        'Argument yhatss must be a tuple of the form (model, tensor), or a list'
#    assert (isinstance(yhatss[0], nn.Module) and\
#        isinstance(yhatss[1], torch.Tensor)), dedent("""\
#            If argument yhatss is an iterable, then the first item should be
#            the model, and the second should be the xss.""")
#    model = yhatss[0].to(device)
#    with torch.no_grad():
#      yhatss = model(yhatss[1].to(device))
#  assert yhatss.dim() == yss.dim(), dedent("""\
#      The arguments yhatss (dim = {}) and yss (dim = {}) must have the
#      same dimension.""".format(yhatss.dim(), yss.dim()))
#  assert yhatss.dim() == 2, dedent("""\
#      Multiple outputs not implemented yet; yhatss should have dimen-
#      sion 2, not {}.""".format(yhatss.dim()))
#  assert len(yhatss) == len(yss), dedent("""\
#      len(yhatss) is {} which is not equal to len(yss) which is {}
#  """.format(len(yhatss),len(yss)))
#  assert yhatss.size()[1] ==  yss.size()[1] == 1, dedent("""\
#      The first dimension of yhatss and yss should index the examples.""")
#  ave_sum_squares = nn.MSELoss()
#  yhatss = yhatss.squeeze(1).to(device)
#  yss = yss.squeeze(1).to(device)
#  SS_E = len(yss) * ave_sum_squares(yhatss, yss)
#  SS_T=len(yss)*ave_sum_squares(yss,yss.mean(0)*torch.ones(len(yss)).to(device))
#  if return_error: return (SS_E/SS_T).item()
#  else: return 1.0-(SS_E/SS_T).item()

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


######## Stuff below is likely obsolete.

#def _batcher(data_tuple, bs, data_dev, model_dev):
#  """Helper function for the train function that returns a gen-
#  erator which, after the data are coherently randomized, kicks
#  out batches of the specified size.
#
#  Args:
#    $data_tuple$ (`Tuple[tensor]`): The tensors to be coherent-
#        ly batched.
#    $bs$ (`int`): The batchsize.
#    $data_dev$ (`Union[str, torch.device]`): The device on which
#        to batch the tensors from.
#    $model_dev$ (`Union[str, torch.device]`): The device to move
#        the batches to just before yielding them.
#
#  Returns:
#    `generator`. A generator that yields batches in the form of
#        tuples of the same length as `data_tuple`.
#  """
#  num_examples = len(data_tuple[0])
#  tuple([t.to(data_dev) for t in data_tuple])
#  indices = torch.randperm(num_examples, device = data_dev)
#  for idx in range(0, num_examples, bs):
#    yield tuple([t.index_select(0,indices[idx: idx + bs]).to(model_dev)\
#        for t in data_tuple])

#def _parse_data(data_tuple, device = 'cpu'):
#  """Simple helper function for the train function.
#
#  Args:
#    $data_tuple$ (`Tuple[tensor]`): Length either 2 or 3.
#
#  Returns:
#    `Tuple[tensor]`.
#  """
#  feats = data_tuple[0].to(device); targs = data_tuple[-1].to(device)
#  if len(data_tuple) == 3:
#    feats_lengths = data_tuple[1].to(device)
#    assert len(feats_lengths) == len(feats),\
#        "No. of feats lengths ({}) must equal no. of feats ({}).".\
#            format(len(feats_lengths), len(feats))
#  else:
#    assert len(data_tuple) == 2, 'data_tuple must have len 2 or 3'
#    feats_lengths = None
#  assert len(feats) == len(targs),\
#      "Number of features ({}) must equal number of targets ({}).".\
#          format(len(feats), len(targs))
#  return feats, feats_lengths, targs

# this is likely obsolete now
#def _explained_var(yss_train, yss_test=None, gpu = (-1,)):
#  """helper to compute explained variation (i.e., variance).
#
#  This returns two functions each of which actually return the
#  average unexplained variance. So, each of the returned func-
#  tions must be de-averaged and then adjusted.  For example,
#  to get the explained variation for the training data, one
#  computes:
#
#    1-len(yss_train)*_explained_var(yss_train,yss_test)[0]
#
#  Something like this is necessary if one wants to compute the
#  explained variance on dataloaders (so batchwise, in an online
#  fashion).
#
#  Note: this is not that useful as a metric.
#
#  But, if you really want to look at this, then call the `train`
#  function like this:
#
#  `model = train(`
#      ...
#      `valid_metric = _explained_var(yss_train, yss_test),`
#      ...
#  `)`
#
#  Args:
#    $yss_train$ (`tensor`)
#    $yss_test$ (`tensor`) Default: `None`.
#
#  Returns:
#    `(function, function)` where each function maps a pair of
#        tensors to a float.
#  """
#  yss_train = yss_train.to(du.utils.get_device(gpu[0]))
#  yss_test = yss_test.to(du.utils.get_device(gpu[-1]))
#  train_fn = lambda yhatss, yss: _sum_square_div(
#      yhatss, yss, _sum_square_div(yss_train, yss_train.mean(0)))
#  #train_fn = lambda yhatss, yss: (yhatss*yhatss).sum()/(yss_train*yss_train).sum()
#  if yss_test is not None:
#    test_fn = lambda yhatss, yss: _sum_square_div(
#        yhatss, yss, _sum_square_div(yss_test, yss_test.mean(0)))
#    #test_fn = lambda yhatss, yss: (yhatss*yhatss).sum()/(yss_test*yss_test).sum()
#  else:
#    test_fn = None
#  return train_fn, test_fn

# this is likely obsolete now
#def _sum_square_div(yhatss, yss, denom=1.0):
#  """Return sum_squared diffs divided by denom.
#
#  Args:
#    $yhatss$ (`Tensor`).
#    $yss$ (`Tensor`).
#    $denom$ (`float`). Default: `1.0`.
#
#  Returns:
#    `float`.
#
#  Examples:
#  >>> _sum_square_div(torch.arange(4.),2*torch.arange(4))
#  tensor(14.)
#
#  >>> yhatss = torch.arange(5.).view(5,1)
#  >>> _sum_square_div(yhatss, yhatss.mean(0))
#  tensor(10.)
#
#  """
#  diffs = yhatss - yss
#  return (diffs * diffs).sum() / denom
