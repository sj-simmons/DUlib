#!/usr/bin/env python3
'''core functions for working with neural nets.

This library contains functions for centering and normalizing
data, splitting out testing data, training neural nets, and
gauging accuracy of trained models.
                   _____________________

The functions along with their signatures displaying the simp-
lest way to call them are (see each functions actual document-
ation for more details):

center
  (xss: tensor) -> Tuple[tensor, tensor]
coh_split
  (proportion: float, *args: tensor) -> Tuple[tensor]
confusion_matrix
  (prob_dists: tensor, yss: tensor, classes: tensor)
cross_validate
  (model, crit, train_data:Tuple[tensor], k:int, bail_after:int)
cross_validate_train
  (model, crit, train_data: Tuple[tensor], k: int)
get_device
  ()
normalize
  (xss: tensor)
optimize_ols
  (feats: tensor)
r_squared
  (yhatss: tensor, yss: tensor)
stand_args
  ()
train
  (model, crit, train_data: Tuple[tensor))
                   _____________________

Todo (you can safely ignore this, unless you want to help
    Simmons work on these items.)
  - fix the packing issue for minibatch in rec nets. Graphing
    against test loss on rec nets doesn't naturally work until
    then.
  - attempt to move to device only in train() and coh_split().
    So try to refrain to going to device in programs (but still
    get and pass the device, of course).
  - make cross_validate_train and cross_validate work with
    variable length data.
  - add feats lengths to all three train fns and documentation.
  - zparams in train() should probably be a tensor on the same
    device as the model parameters.
  - Add option to train to show progress on training / testing
    data each epoch.  Done for losses, but add another pane
    to the graph with percentage correct training/testing.
  - Add percentage or loss to ascii output in the presence of
    testing data. (Take into account forget_after here).
  - Implement stratified sampling at least when splitting out
    testing data.  Maybe pass the relevant proportions to
    coh_split.
  - Try to catch last saved model or just continue on control-c
    for, well, everything.
    - Fix catch_sigint_and_break or remove it. If it is working
      in bash, check and see how it interacts with interrupt in
      say IDLE.
  - Allow/implement adaptive momentum and activate train
    accordingly.
  - Clean up verbosity in cross_validate_train and
    cross_validate.
  - Start type checking kwargs whenever everyone is clearly
    running Python 3.6 or greater.
  - break out processing of train_data test_data in train() as
    a helper fn. Write a helper called to_device().
  - clean up string by using multiline ones correctly. Use
    verbosity so nothing is printed by default?
'''
import torch
import torch.nn as nn

__author__ = 'Simmons'
__version__ = '0.6'
__status__ = 'Development'
__date__ = '11/17/19'

def get_device(gpu = -1):
  '''Get the best device to run on.

  Args:
    gpu (int): The gpu to use. Set to -1 to use the last gpu
        found when gpus are present; set to -2 to override
        using a found gpu and use the cpu. Default -1.

  Returns:
    str. A string that can be passed using the  `to` method
        of Torch tensors and modules.
  '''
  if gpu > -2:
    return torch.device( "cuda:{0}".format(
               (torch.cuda.device_count() + gpu) % torch.cuda.device_count()
           )) if torch.cuda.is_available() else "cpu"
  else:
    return 'cpu'

def center(xss, new_centers = None):
  '''(Mean, by default) center a tensor.

  With this you can translate the data to anywhere. If the
  second argument is `None`, then this simply mean-centers the
  data w/r to the first dimension. But notice that the return-
  ed object is a tuple. So if you want to simply mean-center a
  tensor you would call this function like:

  xss_centered, _ = center(xss)

  Examples:

  >>> xss = torch.arange(12.).view(3,4)
  >>> center(xss)
  (tensor([[-4., -4., -4., -4.],
          [ 0.,  0.,  0.,  0.],
          [ 4.,  4.,  4.,  4.]]), tensor([4., 5., 6., 7.]))
  >>> xss_, xss_means =  center(xss)
  >>> xss__, _ = center(xss_, -xss_means)
  >>> int(torch.all(torch.eq(xss, xss__)).item())
  1

  Args:
    xss (torch.Tensor) The tensor to center.
    new_centers(torch.Tensor) A tensor the number of dimensions
        of which is one less than that of `xss` and whose size
        is in fact `torch.Size([d_1,...,d_n])` where `xss` has
        `torch.Size([d_0, d_1,...,d_n])`.  The default `None`
        is equivalent to `new_center` being the zero tensor.
        Default: None.

  Returns:
    (torch.Tensor, torch.Tensor). A tuple of tensors the first
        of which is xss centered with respect to the first dim-
        ension; the second is a tensor the size of the remain-
        ing dimensions that holds the means.
  '''
  # add and assert here ... check that new_center is right dim.
  xss_means = xss.mean(0)
  if isinstance(new_centers, torch.Tensor):
    new_xss = xss.sub_(new_centers)
  else:
    new_xss = xss.sub_(xss_means)
  return new_xss, xss_means

def normalize(xss, new_widths = None, unbiased = True):
  '''Normalize without dividing by zero.

  See the documentation for the function `center`. This is
  completely analagous.

  Examples:

  >>> xss = torch.tensor([[1, 2, 3], [6, 7, 8]]).float()
  >>> xss, _ = normalize(xss, unbiased = False)
  >>> xss.tolist() # doctest:+ELLIPSIS
  [[0.4...
  >>> xss = torch.tensor([[1, 2, 3], [1, 7, 3]]).float()
  >>> xss, _ = normalize(xss, unbiased = False)
  >>> xss.tolist() # doctest:+ELLIPSIS
  [[1.0...

  Args:
    xss (torch.Tensor)
    new_widths (torch.Tensor)
    unbiased (bool): If unbiased is False, divide by n instead
        of (n-1) when computing the standard deviation.

  Returns:
    (torch.Tensor, torch.Tensor). A tuple of tensors the first
        of which is xss normalized with respect to the first
        dimension, except that those columns with standard dev
        less than a threshold are left unchanged. The list of
        standard devs, with numbers less than the threshold
        replaced by 1.0, is the second tensor returned.
  '''
  # add and assert here ... check that new_width is right dim.
  xss_stdevs = xss.std(0, unbiased)
  xss_stdevs[xss_stdevs < 1e-7] = 1.0
  if isinstance(new_widths, torch.Tensor):
    new_xss = xss.div_(new_widths)
  else:
    new_xss = xss.div_(xss_stdevs)
  return new_xss, xss_stdevs

def coh_split(proportion, *args, device = 'cpu'):
  '''Coherently randomize and split tensors into training and
  testing tensors.

  This splits with respect to the first dimension.

  Args:
    proportion (float): The proportion to split out. Suppose
        this is 0.8. Then for each pair in the return tuple,
        the first holds 4/5 of the data and the second holds
        the other 1/5.
    *args (torch.tensor): The tensors to be randomized and
        split, which must each have a common length in the
        first dimension.
    device (str): The returned tensors have been sent to
        this device.

  Returns:
    Tuple(torch.tensor). A tuple of length twice that of `args`
        and holding, in turn, pairs, each of which is a tensor
        in `args` split according to `proportion` and sent to
        the specified `device`.

  Examples:

  >>> from torch import rand
  >>> coh_split(0.6, rand(2,3), rand(3,3))
  Traceback (most recent call last):
    ...
  AssertionError: all tensors must have same size in first dim
  >>> xss=rand(4, 2); xss_lengths=rand(4); yss=rand(4, 3)
  >>> len(coh_split(0.6, xss, xss_lengths, yss))
  6
  >>> xss_train, xss_test, *_ = coh_split(0.75, xss, yss)
  >>> xss_train.size()
  torch.Size([3, 2])
  '''
  assert 0 <= proportion <= 1, "proportion ({}) must be between 0 and 1, "+\
      "inclusive".format(proportion)
  len_ = list(map(len, args))
  assert all(len_[0] == x for x in len_), "all tensors must have same size "+\
      "in first dim"
  indices = torch.randperm(len_[0]).to(device)
  rand_args = [tensor.to(device).index_select(0, indices) for tensor in args]
  cutoff = int(proportion * len_[0])
  split_args = [[tensor[:cutoff], tensor[cutoff:]] for tensor in rand_args]
  return_args =[item for sublist in split_args for item in sublist]
  return tuple(return_args)

def train(model, crit, train_data, **kwargs):
  '''Train a model.

  Assuming that the number of training examples is divisible
  by the batchsize, the loss printed is the average loss per
  sample over each epoch training. (Under the same assumption,
  one epoch of training corresponds to the model seeing each
  example in the training data exactly once.

  If graphing is enabled, the black graph is the (average)
  loss for the training data. If testing data is present,
  then the red graph is the average loss per example for the
  test data.

  Args:
    model (nn.Module): The instance of Module to be trained.
    crit (nn.modules.loss): The loss function when training.
    train_data Tuple[torch.Tensor]: A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tensor is
        only necessary when training a recurrent net on var-
        iable length inputs. In that case, the triple of ten-
        sors must be of the form
          (train_features, train_lengths, train_targets).
        That is, the first tensor holds the inputs of the
        training data, the second holds the corresponding
        lengths, and the third holds the training data out-
        puts.
        If the data are not of variable length, then there
        is no need to pass the middle tensor in the triple
        above. So one passes
                 (train_features, train_targets).
        In any case, each of the tensors in the tuple must
        be of dimension at least two, with the first dimen-
        sion indexing the training examples.

  Kwargs:
    test_data Tuple[torch.Tensor]: Data to test on in the
        form of a tuple of length 2 or 3 (i.e., matching
        the `train_data` (see above).  If present, the
        loss on test data is computed each epoch.  However,
        The test data is not shown to the model during as
        part of backpropagation. Default = None.
    graph (int): If positive then, during training, display
        a real-time graph.  If greater than 1, then the be-
        gining `graph` losses are thrown away when training
        gets to epoch `graph` (this functionality is made
        available for a better viewing experience for some
        models). Requires matplotlib (and a running X server).
        If 0, do not display a graph. Default: 0.
    feats_lengths (torch.LongTensor): One-dimensional tensor
        holding the lengths of sequences in `feats`. Likely,
        relevant only for variable-length (i.e,, sequence)
        features. Default: None.
    lr (float): The learning rate to be used during training.
        Default: 0.1.
    mo (float): The momentum during training. Default: 0.0.
    bs (int): The mini-batch size where -1 forces batch grad-
        ient descent (i.e. feed-forwarding all training exam-
        ples before each backpropagation). Default: -1.
    eps (int): The number of epochs to train over, where an
        epoch is duration required to see each training ex-
        ample exactly once. Default: 10.
    adapts (Dict): A dictionary mapping each of (or at least
        one of) the strings 'lr', 'mo' to a lambda function
        which itself maps a float to a float. The lambda fns
        will be applied before each backpropagation.  E.g.,
        {'lr': lambda x: 0.98*x} corresponds to learning
        rate decay. Default: the identity map(s).
    print_lines (Tuple[int, int]): A tuple, the first compon-
        ent of which is the number of lines to print initial-
        ly when printing the current loss for each epoch dur-
        ing training, and the second of which is the number
        of lines to print lastly when training. If at least
        one element of the tuple is 0 (resp., -1), then no
        (resp., all) lines are printed. Default: (17, 7).
    verb (int): The verbosity. 0: silent, ... , 2: all.
        Default: 2.
    device (str): The device to run on. Default: 'cpu'.

  Returns:
    (nn.Module, Tuple). The trained model.
  '''
  _catch_sigint()

  test_data = kwargs.get('test_data', None)
  lr = kwargs.get('lr', 0.1); mo = kwargs.get('mo', 0.0);
  bs=kwargs.get('bs', -1); eps = kwargs.get('eps', 10)
  adaptives = kwargs.get('adapts', {'lr': lambda x: x, 'mo': lambda x: x})
  print_init, print_last = kwargs.get('print_lines',(8, 12))
  verb = kwargs.get('verb', 2); device = kwargs.get('device', 'cpu')
  graph = kwargs.get('graph', 0)

  assert graph>=0, 'graph must be a non-negative integer, not {}.'.format(graph)

  # parse train_data
  train_feats=train_data[0].to(device); train_targs=train_data[-1].to(device)
  if len(train_data) == 3:
    train_feats_lengths = train_data[1].to(device)
    assert len(train_feats_lengths) == len(train_feats),\
        "No. of train_feats lengths ({}) must equal no. of train_feats ({}).".\
            format(len(train_feats_lengths), len(train_feats))
  else:
    assert len(train_data) == 2, 'test data must have len 2 or 3'
    train_feats_lengths = None
  assert len(train_feats) == len(train_targs),\
      "Number of training features ({}) must equal number of targets ({}).".\
          format(len(train_feats), len(train_targs))

  model = model.to(device)
  train_feats = train_feats.to(device); train_targs = train_targs.to(device)
  if isinstance(train_feats_lengths, torch.Tensor):
    train_feats_lengths = train_feats_lengths.to(device)

  num_examples = len(train_feats)

  # parse test_data
  if test_data:
    losses_test = []
    test_feats = test_data[0].to(device); test_targs = test_data[-1].to(device)
    if len(test_data) == 3:
      test_feats_lengths = test_data[1].to(device)
      assert len(test_feats_lengths) == len(test_feats),\
          "No. of test_feats lengths ({}) must equal no. of test_feats ({}).".\
              format(len(test_feats_lengths), len(test_feats))
    else:
      assert len(test_data) == 2, 'test_data tuple must have len 2 or 3'
      test_feats_lengths = None
    assert len(test_feats) == len(test_targs),\
        "Number of testing features ({}) must equal number of targets ({}).".\
            format(len(test_feats), len(test_targs))

    test_feats = test_feats.to(device); test_targs = test_targs.to(device)
    if isinstance(test_feats_lengths, torch.Tensor):
      test_feats_lengths = test_feats_lengths.to(device)

  if bs <= 0: bs = num_examples
  if  print_init == -1 or print_last == -1: print_init, print_last = eps, -1

  if mo > 0.0:
    z_params = []
    for param in model.parameters():
      z_params.append(param.data.clone())
    for param in z_params:
      param.zero_()

  if verb > 2: print(model)

  if verb > 1:
    if lr < .005: lr_str = "learning rate: {:.4g}".format(lr)
    else: lr_str = "learning rate: {:.5g}".format(lr)
    if mo < .005: mo_str = ", momentum: {:.4g}".format(mo)
    else: mo_str = ", momentum: {:.5g}".format(mo)
    print(lr_str + mo_str + ", batchsize:", bs)

  if graph:
    import matplotlib.pyplot as plt
    plt.ion(); fig, _ = plt.subplots()
    plt.xlabel('epoch',size='larger'); plt.ylabel('average loss',size='larger')

  losses = []

  for epoch in range(eps):
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
      model.zero_grad()
      loss.backward()

      if mo > 0.0:
        for i, (z_param, param) in enumerate(zip(z_params, model.parameters())):
          z_params[i] = mo * z_param + param.grad.data
          param.data.sub_(z_params[i] * lr)
      else:
        for param in model.parameters():
          param.data.sub_(lr * param.grad.data)

      lr = adaptives['lr'](lr) # apply adaptives

    if print_init * print_last != 0 and verb > 0:
      loss_len = 20
      base_str = "epoch {0}/{1}; loss ".format(epoch+1, eps)
      loss_str = "{0:<10g}".format(accum_loss*bs/num_examples)
      if eps < 20 or epoch < print_init:
        print(base_str + loss_str)
      elif epoch > eps - print_last:
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
      if  epoch == graph:
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

  return model

def cross_validate_train(model, crit, train_data, k, **kwargs):
  '''Cross-validate train a model for one (by default) epoch.

  Rather than calling this directly, consider calling the
  function `cross_validate` in this module.

  Args:
    model (nn.Module): The instance of Module to be trained.
    crit (nn.modules.loss): The loss function when training.
    train_data Tuple[torch.Tensor]: A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tensor is
        only necessary when training a recurrent net on var-
        iable length inputs. In that case, the triple of ten-
        sors must be of the form
          (train_features, train_lengths, train_targets).
        That is, the first tensor holds the inputs of the
        training data, the second holds the corresponding
        lengths, and the third holds the training data out-
        puts.
        If the data are not of variable length, then there
        is no need to pass the middle tensor in the triple
        above. So one passes
                 (train_features, train_targets).
        In any case, each of the tensors in the tuple must
        be of dimension at least two, with the first dimen-
        sion indexing the training examples.
    k (int): The number of folds on which to cross-validate.
        Default: 10.

  Kwargs:
    valid_crit (nn.Module):  The validation criterion to use
        when gauging the accuracy of the model on test data.
        If None, this is set to `crit`; i.e., the training
        criterion. Default: None.
    cent_norm_feats (Tuple[bool]): Tuple with first entry det-
        ermining whether to center the features, and the sec-
        ond, whether to normalize them. Default: (True, True).
    cent_norm_targss (Tuple[bool]): Tuple with first entry det-
        ermining whether to center the targets, and the sec-
        ond, whether to normalize them. Default: (True, True).
    feats_lengths (torch.LongTensor): One-dimensional tensor
        holding the lengths of sequences in `feats`. Likely,
        relevant only for variable-length (i.e,, sequence)
        features. Default: None.
    lr (float): The learning rate to be used during training.
        Default: 0.1.
    mo (float): The momentum during training. Default: 0.0.
    bs (int): The mini-batch size where -1 forces batch grad-
        ient descent (i.e. feed-forwarding all training exam-
        ples before each backpropagation). Default: -1.
    eps (int): The number of epochs to train over for each
        validation step. Default: 1.
    adapts (Dict): A dictionary mapping each of (or at least
        one of) the strings 'lr', 'mo' to a lambda function
        which itself maps a float to a float. The lambda fns
        will be applied before each backpropagation.  E.g.,
        {'lr': lambda x: 0.98*x} corresponds to learning
        rate decay. Default: the identity map(s).
    verb (int): The verbosity. 0: silent, ... , 2: all.
        Default: 2.
    device (str): The device to run on. Default: 'cpu'.

  Returns:
    nn.Module. Returns the model which has been partially
        trained (for one epoch, by default) along with a ten-
        sor of its validations. The data are appropriately
        centered and normalized during training. If the num-
        ber of the features is not divisible by k, then the
        last chunk is thrown away (so make the length of it
        small, if not zero).
  '''
  catch_sigint()
  valid_crit = kwargs.get('valid_crit', None)
  feats, targs = train_data
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  feats_lengths = kwargs.get('feats_lengths', None)
  cent_feats, norm_feats = kwargs.get('cent_norm_feats',(True, True))
  cent_targs, norm_targs = kwargs.get('cent_norm_targs',(True, True))
  lr = kwargs.get('lr', 0.1); mo = kwargs.get('mo', 0.0);
  bs=kwargs.get('bs', -1); eps = kwargs.get('eps', 1)
  adapts = kwargs.get('adapts', {'lr': lambda x: x, 'mo': lambda x: x})
  verb = kwargs.get('verb', 2); device = kwargs.get('device', 'cpu')

  valids = torch.zeros(k) # this will hold the k validations
  chunklength = len(feats) // k

  if not valid_crit: valid_crit = crit

  # randomize
  indices = torch.randperm(len(feats)).to(device)
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

    model = train(model=model, crit=crit, train_data=(xss_train, yss_train),
        eps=eps, lr=lr, mo=mo, bs=bs, adapts=adapts, verb=verb-1, device=device)

    if cent_feats: xss_test.sub_(xss_train_means)
    if norm_feats: xss_test.div_(xss_train_stdevs)
    if cent_targs: yss_test.sub_(yss_train_means)
    if norm_targs: yss_test.div_(yss_train_stdevs)

    valids[idx//chunklength] = valid_crit(model(xss_test), yss_test)

  return model, valids

def cross_validate(model, crit, train_data, k, bail_after, **kwargs):
  '''Cross-validate a model.

  Args:
    model (nn.Module): The instance of Module to be trained.
    crit (nn.modules.loss): The loss function when training.
    train_data Tuple[torch.Tensor]: A tuple consisting of
        either 2 or 3 tensors. Passing a length 3 tensor is
        only necessary when training a recurrent net on var-
        iable length inputs. In that case, the triple of ten-
        sors must be of the form
          (train_features, train_lengths, train_targets).
        That is, the first tensor holds the inputs of the
        training data, the second holds the corresponding
        lengths, and the third holds the training data out-
        puts.
        If the data are not of variable length, then there
        is no need to pass the middle tensor in the triple
        above. So one passes
                 (train_features, train_targets).
        In any case, each of the tensors in the tuple must
        be of dimension at least two, with the first dimen-
        sion indexing the training examples.
    k (int): The number of folds on which to cross-validate.
        Default: 10.
    bail_after (int): The number of steps of cross_validation
        training after which to bail if no improvement is
        seen. Default: 10.

  Kwargs:
    valid_crit (nn.Module):  The validation criterion to use
        when gauging the accuracy of the model on test data.
        If None, this is set to `crit`; i.e., the training
        criterion. Default: None.
    cent_norm_feats (Tuple[bool]): Tuple with first entry det-
        ermining whether to center the features, and the sec-
        ond, whether to normalize them. Default: (True, True).
    cent_norm_targss (Tuple[bool]): Tuple with first entry det-
        ermining whether to center the targets, and the sec-
        ond, whether to normalize them. Default: (True, True).
    feats_lengths (torch.LongTensor): One-dimensional tensor
        holding the lengths of sequences in `feats`. Likely,
        relevant only for variable-length (i.e,, sequence)
        features. Default: None.
    lr (float): The learning rate to be used during training.
        Default: 0.1.
    mo (float): The momentum during training. Default: 0.0.
    bs (int): The mini-batch size where -1 forces batch grad-
        ient descent (i.e. feed-forwarding all training exam-
        ples before each backpropagation). Default: -1.
    eps (int): The number of epochs to train over for each
        validation step. Default: 1.
    adapts (Dict): A dictionary mapping each of (or at least
        one of) the strings 'lr', 'mo' to a lambda function
        which itself maps a float to a float. The lambda fns
        will be applied before each backpropagation.  E.g.,
        {'lr': lambda x: 0.98*x} corresponds to learning
        rate decay. Default: the identity map(s).
    verb (int): The verbosity. 0: silent, ... , 2: all.
        Default: 2.
    device (str): The device to run on. Default: 'cpu'.

  Returns:
    nn.Module. Returns the model which has been partially
        trained (for one epoch, by default) along with a ten-
        sor of its validations. The data are appropriately
        centered and normalized during training. If the num-
        ber of the features is not divisible by k, then the
        last chunk is thrown away (so make the length of it
        small, if not zero).

  '''
  import copy

  valid_crit = kwargs.get('valid_crit', None)
  feats, targs = train_data
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  feats_lengths = kwargs.get('feats_lengths', None)
  bail_after = kwargs.get('bail_after', 10)
  cent_feats, norm_feats = kwargs.get('cent_norm_feats',(True, True))
  cent_targs, norm_targs = kwargs.get('cent_norm_targs',(True, True))
  lr = kwargs.get('lr', 0.1); mo = kwargs.get('mo', 0.0);
  bs=kwargs.get('bs', -1); eps = kwargs.get('eps', 1)
  adapts = kwargs.get('adapts', {'lr': lambda x: x, 'mo': lambda x: x})
  verb = kwargs.get('verb', 2); device = kwargs.get('device', 'cpu')

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
        train_data = (feats, targs),
        k = k,
        valid_crit = valid_crit,
        eps = eps,
        lr = lr,
        mo = mo,
        bs = bs,
        verb = verb,
        device = device)

    total_epochs += k*eps

    if valids.mean().item() < best_valids.mean().item():
      best_model = copy.deepcopy(model)
      best_valids = valids
      no_improvement = 0
    else:
      no_improvement += 1

    if valids.mean().item() == 0.0: no_improvement = bail_after

    if verb > 0:
      print("epoch {3}; valids: mean={0:<7g} std={1:<7g}; best={2:<7g}"\
          .format(valids.mean().item(),valids.std().item(),best_valids.mean().\
          item(),total_epochs)+' '+str(no_improvement)+"/"+str(bail_after))

  if verb > 0:
    print("best valid:  mean={0:.5g}  stdev={1:.5g}"\
        .format(best_valids.mean().item(),best_valids.std().item()))

  return best_model, best_valids.mean()

def optimize_ols(feats, **kwargs):
  '''Compute the optimal learning rate and, optionally, momen-
  tum.

  The returned values are only optimal (or even relevant) for
  linear regression models; i.e. for linear models with MSE
  loss.

  Reports the condition number of A = X^T*X where X is the des-
  ign matrix.

  Args:
    feats (torch.Tensor): The features of the training data.

  Kwargs:
    with_mo (bool): Optimize both the learning rate and the
        momentum. Default: True.
    warn (bool): Issue warnings about numerical integrity.
        Default: True.
    verb (int): Put 0 for silent. Default: 1.

  Returns:
    Tuple[float]: A tuple, the first entry of which is the
        optimal learning_rate and second of which is the
        optimal momentum. If `with_mo` is False, then 0.0
        is returned for the momentum.
  '''
  from scipy.linalg import eigh
  from scipy.sparse.linalg import eigsh
  from scipy.sparse import issparse

  with_mo = kwargs.get('with_mo', True)
  warn = kwargs.get('warn', True)
  verb = kwargs.get('verb', 1)

  problematic = False
  if verb: print("optimizing ... ", end='')

  # ADD A COLUMN OF ONES
  feats = torch.cat((torch.ones(len(feats),1), feats.to("cpu")), 1)

  feats = feats.numpy().astype('float64')
  A = feats.transpose() @ feats
  eigs = eigh(A, eigvals_only = True)
  if not all(map(lambda x: x.imag == 0.0, eigs)) == True and warn and verb:
    print("\nwarning: eigenvalues should be real but some are not due to num"+\
        "erical\n"+' '*9+"ill-conditioning (largest imaginary part is "+\
        '{:.3g}'.format(max([x.imag for x in eigs]))+").")
    problematic = True
  eigs = [x.real for x in eigs]
  if not all(map(lambda x: x >= 0.0, eigs)) == True and warn and verb:
    print("\nwarning: eigenvalues should be positive but some are not due to "+\
        "numerical\n"+' '*9+"ill-conditioning (most negative eigenvalue is "+\
        '{:.3g}'.format(min([x for x in eigs]))+").")
    problematic = True

  if problematic:
    if verb: print("checking for sparseness ... ",end='')
    is_sparse = issparse(A)
    if verb: print(sparse)
    largest = eigsh(A,1,which = 'LM',return_eigenvectors = False).item()
    smallest=eigsh(A,1,which='SA',return_eigenvectors=False,sigma=1.0).item()
  else:
    eigs = [0.0 if x.real < 0.0 else x for x in eigs]
    largest = max(eigs)
    smallest = min(eigs)

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

  return learning_rate, momentum

def confusion_matrix(prob_dists, yss, classes, **kwargs):
  '''Compute the confusion matrix.

  Compute the confusion matrix with respect to given prob_dists
  and targets.  The columns in the displayed table correspond
  to the actual (correct) target class and the rows are the
  class predicted by model.

  Args:
    prob_dists (torch.Tensor): A tensor of dimension 2 hold-
        ing the probability distribution predicting the cor-
        rect class for each example. The first dimension must
        index the examples. This is the predictions, in the
        form of probability distributions, made by a model
        when fed the features of some set of examples.
    yss (torch.Tensor): A 1-dimensional tensor holding the
        correct class for each example.
    classes (torch.LongTensor): A one-dimensional tensor
        holding the numerical version='0.6',
        torch.arange(10) for digit classification.

  Kwargs:
    return_error (bool): If True return return the error in
        the form of a float between 0 and 1 inclusive rep-
        resenting the error; if False, return a float rep-
        resenting the proportion of examples correctly class-
        ified. Default: False.
    show (bool): If True show display the confusion matrix.
        Default: False.
    class2name (Dict[int, str]): A dictionary mapping each
        numerical class to its classname. Default: None.

  Returns:
    float.  The total proportion (a number between 0 and 1)
        of correct correct predictions or (optionally) one
        minus that ratio; i.e., the error rate.
  '''
  assert len(prob_dists) == len(yss),\
      'Number of features ({}) must equal number of targets ({}).'\
          .format(len(prob_dists), len(yss))
  assert prob_dists.dim() == 2,\
      'The prob_dists argument should be a 2-dim tensor not a {}-dim one.'\
          .format(prob_dists.dim())
  assert classes.dim() == 1,\
      'The classes argument should be a 1-dim tensor not {}-dim one.'\
          .format(classes.dim())

  return_error = kwargs.get('return_error', False)
  show = kwargs.get('show', False)
  class2name = kwargs.get('classnames', None)

  cm_counts = torch.zeros(len(classes), len(classes))
  for prob, ys in zip(prob_dists, yss):
    cm_counts[torch.argmax(prob).item(), ys] += 1

  cm_pcts = cm_counts/len(yss)
  counts = torch.bincount(yss)

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
      for entry in row:
        if entry == 0.0:
          print((cell_length-1)*' '+'0', end='')
        elif entry == 100.0:
          print((cell_length-3)*' '+'100', end='')
        else:
          string = '{:.1f}'.format(100*entry).lstrip('0')
          print(' '*(cell_length-len(string))+string, end='')
      n_examples = cm_counts[:,i].sum()
      pct = 100*(cm_counts[i,i]/n_examples)
      if class2name:
        print('  {} ({:.1f}% of {})'.format(class2name[i],pct,int(counts[i])))
      else:
        print(' ({:.1f}% of {})'.format(pct, int(counts[i])))

  if return_error:
    return 1-torch.trace(cm_pcts).item()
  else:
    return torch.trace(cm_pcts).item()

def r_squared(yhatss, yss, return_error = False ):
  '''
  Returns the coefficient of determination of two 2d tensors
  (where the first dimension in each indexes the examples),
  one holding the yhatss (the predicted outputs) and the other
  holding the actual outputs, yss.

  Note: this is rigorously relevant only to linear, meaning
  possibly polynomial linear, regression.

  Args:
    yhatss (torch.Tensor): The predicted outputs.
    yss (torch.Tensor): The actual outputs.
    return_error (bool): If False return the proportion of the
        variation explained by the regression line. If True,
        return 1 minus that proportion. Default: False.
  '''
  SS_E, SS_T = 0.0, 0.0
  mean = yss.mean(0)
  for yhats, ys in zip(yhatss, yss):
    SS_E += (ys - yhats).pow(2)
    SS_T += (ys - mean).pow(2)
  if return_error:
    return (SS_E/SS_T).item()
  else:
    return 1.0-(SS_E/SS_T).item()

def stand_args(desc = '', **kwargs):
  '''Set standard hyper-parameters.

  Setup argparse switches, etc. for standard hyper-parameters,
  and return the parser object so the calling program can add
  more switches.

  Note: This function does not implement, for example, bs being
  set to -1 leading to (full) batch gradient descent. That
  needs to be implemented in the program that calls this func-
  tion.  Said differently, this function is handy solely for
  elimination boilerplate processing of the standard hyperpar-
  meters like learning rate, momentum, etc.

  Args:
    desc (str): A short description of what the program does.
        Default: ''

  Kwargs:
    lr (float): The learning rate, returned to the calling
        program via the return parser object with name 'lr'
        and default value `lr`. Default: 0.1.
    mo (float): The momentum returned with name 'mo' and de-
        fault value `mo`. Default 0.0.
    bs (int): Batchsize, where -1 leads to batch gradient
        descent. Default 1.
    eps (int): The number of epochs over which to train.
        Default 20.
    seed (bool): Whether or not to set a random seed.
        Default: False.
    pt (float): The proportion on which train. Default 1.0.
    gpu (int): Which gpu to use in the presence of one or more
        gpus, where -1 means to use the last gpu found, and -2
        means to override using a found gpu and use the cpu.
        Default: -1.
    gr (int): If positive then, during training, display
        a real-time graph.  If greater than 1, then the be-
        gining `graph` losses are thrown away when training
        gets to epoch `graph` (this functionality is made
        available for a better viewing experience for some
        models). Requires matplotlib (and a running X server).
        If 0, do not display a graph. Default: 0.

  Returns:
    (argparse.ArgumentParser). The parser object to which the
        calling program can add more names.
  '''
  import argparse

  desc = kwargs.get('desc', '')
  lr = kwargs.get('lr', 0.1)
  mo = kwargs.get('mo', 0.0)
  bs = kwargs.get('bs', 1)
  eps = kwargs.get('eps', 20)
  seed = kwargs.get('seed', False)
  pt = kwargs.get('pt', 1.0)
  gpu = kwargs.get('gpu', -1)
  gr = kwargs.get('gr', 0)

  parser = argparse.ArgumentParser( description = desc, formatter_class =\
      argparse.ArgumentDefaultsHelpFormatter)
  p = parser.add_argument
  p('-lr', type=float, help='learning rate', default=lr)
  p('-mo', type=float, help='momentum', default=mo)
  hstr='the mini-batch size; set to -1 for (full) batch gradient descent'
  p('-bs', type=int, help=hstr, default=bs)
  p('-eps', type=int, help='num epochs to train', default=eps)
  hstr="toggle setting random seed"
  if seed:
    p('-seed', dest='seed', help=hstr, action='store_false')
  else:
    p('-seed', dest='seed', help=hstr, action='store_true')
  p('-ser', help='serialize the trained model', action='store_true')
  p('-pt', type=float, help='proportion to train on', default=pt)
  hstr='which gpu, if more than one is found; -1 for last gpu found; -2 for cpu'
  p('-gpu', type=int, help=hstr, default=gpu)
  hstr='graph of losses during training; redraw after this many epochs'
  p('-gr', help=hstr, type=int, default=0)
  return  parser

def _catch_sigint():
  '''Catch keyboard interrupt signal.  '''
  import signal
  def keyboardInterruptHandler(signal, frame):
    #print("KeyboardInterrupt (ID: {}) caught. Cleaning up...".format(signal))
    print("\n")
    exit(0)
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

def _catch_sigint_and_break():
  '''Catch keyboard interrupt signal and break out of a `for` or
  `while` loop.
  '''
  import signal
  def keyboardInterruptHandler(signal, frame):
    global interrupted
    interrupted = True
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

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
