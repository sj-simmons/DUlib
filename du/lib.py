#!/usr/bin/env python3
'''Core functions for working with neural nets.

Todo:
  - Try to catch last saved model or just continue on control-c
    for, well, everything.
    - fix catch_sigint_and_break or remove it. If it is working
      in bash, check and see how it interacts with interrupt in
      say IDLE.
  - Implement stratified sampling at least when splitting out
    testing data.  Maybe pass the relevant proportions to
    coherent_split.
  - add option to train to show progress on training / testing
    data each epoch.
  - consider consolidating print_init, print_last into a tuple
    (called printlines)
  - then put None for no printing of lines, thereby cleaning up
    verbosity
'''
import torch
import torch.nn as nn

__author__ = 'Simmons'
__version__ = '0.2'
__status__ = 'Development'
__date__ = '11/6/19'

def get_device(gpu = -1):
  '''Get the best device to run on.

  Args:
    gpu (int): The gpu to use. Set this to -1 to use the last gpu
        found when there is a least one gpu; set this to -2 to
        override using a found gpu and use the cpu.  Default -1.

  Returns:
    str. A string to pass to Torch indicating the best device.
  '''
  if gpu > -2:
    return torch.device( "cuda:{0}".format(
               (torch.cuda.device_count() + gpu) % torch.cuda.device_count()
           )) if torch.cuda.is_available() else "cpu"
  else:
    return 'cpu'

def center(xss):
  '''Mean-center.

  :type xss: torch.Tensor
  :return: Tuple of tensors the first of which is xss mean-centered
      w/r to the first dimension; the second is a tensor the size of
      the remaining dimensions that holds the means.
  '''
  means = xss.mean(0)
  return xss.sub_(means), means

def normalize(xss):
  '''Normalize.

  Normalizes without dividing by zero.  Returns the xss with columns
  normalized except that those columns with standard deviation less
  than a threshold are left unchanged.  The list of standard devs, with
  numbers less than the threshold replaced by 1.0, is also returned.
  '''
  stdevs = xss.std(0)
  stdevs[stdevs < 1e-7] = 1.0
  return xss.div_(stdevs), stdevs

def coherent_split(proportion, *args, device = 'cpu'):
  '''Coherently randomize and split tensors into training and testing
  tensors along the first dimension.

  Args:
    proportion (float): the proportion to split out.
    *args (torch.tensor): The tensors to be randomized and split, all
        of which must have the same length in the first dimension.
  Returns:
    Tuple(torch.tensor). A tuple length twice that of `args`, holding,
        in turn, pairs each tensor in `args` split according to the
        specified `proportion`, and sent to the specified `device`.


  >>> coherent_split(0.6, torch.rand(2,3), torch.rand(3,3))
  Traceback (most recent call last):
    ...
  AssertionError: all tensors must have same size in first dimension
  >>> xss=torch.rand(3, 2); xss_lengths=torch.rand(3); yss=torch.rand(3, 3)
  >>> coherent_split(0.6, xss, xss_lengths, yss) # doctest: +SKIP
  >>> xss = torch.arange(12).view(3,4); yss = torch.arange(3)
  >>> coherent_split(0.8, xss, yss) # doctest: +SKIP
  '''
  assert 0 <= proportion <= 1, "proportion ({}) must be between 0 and 1, "+\
      "inclusive".format(proportion)
  len_ = list(map(len, args))
  assert all(len_[0] == x for x in len_), "all tensors must have same size "+\
      "in first dimension"
  indices = torch.randperm(len_[0])
  rand_args = [tensor.index_select(0, indices) for tensor in args]
  cutoff = int(proportion * len_[0])
  split_args = [[tensor[:cutoff], tensor[cutoff:]] for tensor in rand_args]
  return_args =[item.to(device) for sublist in split_args for item in sublist]
  return tuple(return_args)

def train(model, crit, feats, targs, feats_lengths=None, lr=0.1, mo=0.0, bs=-1,
    eps=10, print_init=7, print_last=17, verb=3, device='cpu'):
  '''Return the model trained with the given hyper-parameters.

  Args:
    model (nn.Module): The instance of nn.Module to be trained.
    crit (nn.modules.loss): The loss function to be used for training.
    feats (torch.Tensor): The training data features (inputs); must
        be a tensor of dimension at least two with the first dimension
        indexing all of the example features for the training data.
    targs (torch.Tensor): The training data outputs; must be a tensor
        with the first dimension indexing the training targets.

  Kwargs:
    feats_lengths (torch.LongTensor): One-dimensional tensor holding
        the lengths of sequences in `feats`. Likely, relevant only
        for variable-length (i.e,, sequence) features. Default: None.
    lr (float): The learning rate during training. Default: 0.1.
    mo (float): The momentum during training. Default: 0.0.
    bs (int): The mini-batch size where -1 means (full) batch gradient
        descent. Default: -1.
    eps (int): The number of epochs to train over. Default: 10.
    print_init (int): The number of loss lines to print initially when
        training (put -1 here for normal printing). Default: 7.
    print_last (int): The number of loss lines to print lastly when
        training.  Default: 17.
    verb (int): Set the verbosity to 0 for no printing while training.
        Default: 3.
    device (str): The device to run on. Default: 'cpu'.

  Returns:
    nn.Module. The trained model.
  '''
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  catch_sigint()
  loss_len = 20
  num_examples = len(feats)
  if bs <= 0: bs = num_examples
  if print_init == -1: print_init = eps

  if mo > 0.0:
    z_params = []
    for param in model.parameters():
      z_params.append(param.data.clone())
    for param in z_params:
      param.zero_()

  if verb > 2:
    print(model)

  if verb > 1:
    if lr < .005: lr_str = "learning rate: {:.4g}".format(lr)
    else: lr_str = "learning rate: {:.5g}".format(lr)
    if mo < .005: mo_str = ", momentum: {:.4g}".format(mo)
    else: mo_str = ", momentum: {:.5g}".format(mo)
    print(lr_str + mo_str + ", batchsize:", bs)

  for epoch in range(eps):
    accum_loss = 0
    indices = torch.randperm(len(feats)).to(device)

    for idx in range(0, num_examples, bs):
      current_indices = indices[idx: idx + bs]

      if isinstance(feats_lengths, torch.Tensor):
        loss = crit(
            model(
                feats.index_select(0, current_indices),
                feats_lengths.index_select(0, current_indices)
            ),
            targs.index_select(0, current_indices)
        )
      else:
        loss = crit(
            model(feats.index_select(0, current_indices)),
            targs.index_select(0, current_indices)
        )
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

    if verb > 0:
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

  return model

def cross_validate_train(
    validation_crit=None,
    k = 10,           # the number of folds
    cent_feats=True,  # whether or not to mean-center the features
    norm_feats=True,  # whether or not to normalize the feature
    cent_targs=True,  # whether or not to mean-center the targets
    norm_targs=True,  # whether or not to normalize the targets
    model=None,       # the model to be trained
    criterion=None,   # the criterion to be used during training
    features=None,    # the inputs of the training data
    targets=None,     # the outputs of the training data
    epochs = 1,       # the number of epochs to train over for each validation
    learning_rate=None,
    momentum = 0.0,
    batchsize = -1,   # -1 means batch gradient descent
    spew_init = 7,    # number of loss lines to display initially when training
    spew_end = 17,    # number of loss lines to display lastly when training
    verbosity = 1,     # verbosity 0 means no printing
    device = 'cpu'
):
  '''
  Returns the model (which has been partially -- for one epoch, by default --
  trained) along with a tensor of its validations.

  Notes:
   -The data are appropriately mean centered and normalized the data during
    training.
   -If the number of the features is not divisible by k, then the last chunk
    is thrown away (so make the length of it small, if not zero).
  '''
  assert len(features) == len(targets),\
      "The number of features must equal number of targets."
  catch_sigint()
  valids = torch.zeros(k) # this will hold the k validations
  chunklength = len(features) // k

  if not validation_crit:
    validation_crit = criterion

  # randomize
  indices = torch.randperm(len(features)).to(device)
  xss = features.index_select(0, indices)
  yss = targets.index_select(0, indices)

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
        model = model,
        crit = criterion,
        feats = xss_train,
        targs = yss_train,
        eps = epochs,
        lr = learning_rate,
        mo = momentum,
        bs = batchsize,
        verb = verbosity - 1
    )

    if cent_feats: xss_test.sub_(xss_train_means)
    if norm_feats: xss_test.div_(xss_train_stdevs)
    if cent_targs: yss_test.sub_(yss_train_means)
    if norm_targs: yss_test.div_(yss_train_stdevs)

    valids[idx//chunklength] = validation_crit(model(xss_test), yss_test)

  return model, valids

def cross_validate(
    validation_crit=None, # the criterion used to validate on testing data
    bail_after = 30,  # bail after this many cross-valid steps is no improvement
    k = 10,           # the number of folds
    cent_feats=True,  # whether or not to mean-center the features
    norm_feats=True,  # whether or not to normalize the feature
    cent_targs=False, # whether or not to mean-center the targets
    norm_targs=False, # whether or not to normalize the targets
    model = None,     # the model to be trained
    criterion=None,   # the criterion to be used during training
    features=None,    # the inputs of the training data
    targets=None,     # the outputs of the training data
    epochs = 1,       # the number of epochs to train over for each validation
    learning_rate=None,
    momentum = 0.0,
    batchsize = -1,   # -1 means (full) batch gradient descent
    spew_init = 7,    # number of loss lines to display initially when training
    spew_end = 17,    # number of loss lines to display lastly when training
    verbosity = 1,     # verbosity 0 means no printing
    device = 'cpu'
):
  import copy
  no_improvement = 0
  best_valids = 1e15*torch.ones(k)
  total_epochs = 0

  if len(features) % k != 0:
    chunklength = len(features) // k
    print("warning: the first",k-1,"chunks have size",chunklength,\
        "but the last one has size",str(len(features) % chunklength)+".")

  if not validation_crit:
    validation_crit = criterion

  while no_improvement < bail_after:

    model, valids = cross_validate_train(
        cent_feats = cent_feats,
        norm_feats = norm_feats,
        cent_targs = cent_targs,
        norm_targs = norm_targs,
        k = k,
        validation_crit = validation_crit,
        model = model,
        criterion = criterion,
        features = features,
        targets = targets,
        epochs = epochs,
        learning_rate = learning_rate,
        momentum = momentum,
        batchsize = batchsize,
        verbosity = verbosity,
        device = device
    )

    total_epochs += k*epochs

    if valids.mean().item() < best_valids.mean().item():
      best_model = copy.deepcopy(model)
      best_valids = valids
      no_improvement = 0
    else:
      no_improvement += 1

    if valids.mean().item() == 0.0: no_improvement = bail_after

    if verbosity > 0:
      print("epoch {3}; valids: mean={0:<7g} std={1:<7g}; best={2:<7g}"\
          .format(valids.mean().item(),valids.std().item(),best_valids.mean().\
          item(),total_epochs)+' '+str(no_improvement)+"/"+str(bail_after))

  if verbosity > 0:
    print("best valid:  mean={0:.5g}  stdev={1:.5g}"\
        .format(best_valids.mean().item(),best_valids.std().item()))

  return best_model, best_valids.mean()

def optimize_ols(
    features: torch.Tensor,
    with_momentum: bool = True,
    warn: bool = True,
    sparse: bool = False
):
  '''
  This only applies to ordinary least squares regression -- in terms of a neural
  net: a linear model with MSE loss.  Returns the optimal learning_rate and
  (optionally, 0.0) momentum. Reports the condition number of A = X^T*X where
  X is the design matrix.
  '''
  from scipy.linalg import eigh
  from scipy.sparse.linalg import eigsh
  from scipy.sparse import issparse


  problematic = False
  print("optimizing ... ", end='')

  # ADD A COLUMN OF ONES
  features = torch.cat((torch.ones(len(features),1), features.to("cpu")), 1)

  features = features.numpy().astype('float64')
  A = features.transpose() @ features
  eigs = eigh(A, eigvals_only = True)
  if not all(map(lambda x: x.imag == 0.0, eigs)) == True and warn:
    print("\nwarning: eigenvalues should be real but some are not due to num"+\
        "erical\n"+' '*9+"ill-conditioning (largest imaginary part is "+\
        '{:.3g}'.format(max([x.imag for x in eigs]))+").")
    problematic = True
  eigs = [x.real for x in eigs]
  if not all(map(lambda x: x >= 0.0, eigs)) == True and warn:
    print("\nwarning: eigenvalues should be positive but some are not due to "+\
        "numerical\n"+' '*9+"ill-conditioning (most negative eigenvalue is "+\
        '{:.3g}'.format(min([x for x in eigs]))+").")
    problematic = True

  if problematic:
    print("checking for sparseness ... ",end='')
    is_sparse = issparse(A)
    print(sparse)
    largest = eigsh(A,1,which = 'LM',return_eigenvectors = False).item()
    smallest=eigsh(A,1,which='SA',return_eigenvectors=False,sigma=1.0).item()
  else:
    eigs = [0.0 if x.real < 0.0 else x for x in eigs]
    largest = max(eigs)
    smallest = min(eigs)

  if (smallest != 0):
    print("condition number: {:.3g}".format(largest/smallest))
  else:
    print("condition number: infinite")

  if not with_momentum:
    learning_rate = 2/(smallest + largest)
    momentum = 0.0
  else:
    learning_rate = (2/(smallest**0.5+largest**0.5))**2
    momentum = ((largest**0.5-smallest**0.5)/(largest**0.5+smallest**0.5))**2

  return learning_rate, momentum

def confusion_matrix(
    yhatss_prob,# the predictions of the model as probability dists
    yss,        # these are the actual targets
    classes,    # a 1d tensor holding the classes; e.g. torch.arange(10)
    return_error = False,
    show = False,
    class2name = None,
):
  '''Compute the confusion matrix.

  Compute the confusion matrix with respect to given yhatss_prob and
  targets.  The columns in the returned tensor or in the diplayed
  table correspond to the actual (correct) target class and the rows
  are the class predicted by model.

  Note: the yhats_prob argument should be roughly model(xss), i.e., (a mini-
  batch of) probability distributions representing the prediction of a model.
  The argument yss holds the corresponding correct values.

  Returns total ratio (a number between 0 and 1) correctly by the model or
  optional one minus that ratio; i.e., the error rate.

  Optionally prints the confusion matrix with percentages.

  Args:
    class2name (Dict[int, str]): A dictionary mapping each numerical
        class to its classname.
  '''
  assert len(yhatss_prob) == len(yss),\
      'Number of features ({}) must equal number of targets ({}).'\
          .format(len(yhatss_prob), len(yss))
  assert classes.dim() == 1,\
      'The classes argument should be a 1-dimensional tensor.'
  cm_counts = torch.zeros(len(classes), len(classes))
  for yhats_prob, ys in zip(yhatss_prob, yss):
    cm_counts[torch.argmax(yhats_prob).item(), ys] += 1

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
          #string = '{:{width}.1f}'.format(100*entry, width=cell_length)
          #print(string)
          #print(type(string))
          #quit()
          #print(string.lstrip('0'), end='')
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
  Returns the coefficient of determination of two 2d tensors (where the
  first dimension indexes the examples) of predicted outputs, yhatss, and
  actual outputs, yss  .

  Note: this is relevant to linear -- meaning polynomial (linear) -- regression.
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

def stand_args(desc='', lr=0.1, mo=0, bs=1, eps=20, seed=False, pt=1, gpu = -1):
  '''Set standard hyper-parameters.

  Setup argparse switches, etc. for standard hyper-paramenters, and
  return the parser object so the calling program can add more switches.

  Args:
    desc (str): A short description of what the program does. Default: ''
    lr (float): The learning rate.
  '''
  import argparse
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
  return  parser

def catch_sigint():
  ''' Catch keyboard interrupt signal.  '''
  import signal
  def keyboardInterruptHandler(signal, frame):
    #print("KeyboardInterrupt (ID: {}) caught. Cleaning up...".format(signal))
    print("\n")
    exit(0)
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

def catch_sigint_and_break():
  ''' Catch keyboard interrupt signal and break out of a `for` or
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


