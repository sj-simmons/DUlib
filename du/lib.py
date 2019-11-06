'''
A library collecting functionality that the DL@DU group has implemented.
'''
import torch
import torch.nn as nn

__author__ = 'Simmons'
__version__ = '0.1'
__status__ = 'Development'
__date__ = "March 2019"

# use (the last) gpu if there are gpus, otherwise use the cpu
device = torch.device("cuda:{0}".format(torch.cuda.device_count()-1)\
    if torch.cuda.is_available() else "cpu")

def center(xss):
  '''
  Return xss mean-centered xss w/r to its first dimension. Also returns the
  means.
  '''
  means = xss.mean(0)
  return xss.sub_(means), means

def normalize(xss):
  '''
  Normalize without dividing by zero.  Returns the xss with columns normalized
  except that those columns with standard deviation less than a  threshold are
  left unchanged.  The list of standard deviations with numbers less than the
  threshold replaced by 1.0 is also returned.
  '''
  stdevs = xss.std(0)
  stdevs[stdevs < 1e-7] = 1.0
  return xss.div_(stdevs), stdevs

def train(
    model,          # the model to be trained
    criterion,      # the criterion to be used during training
    features,       # the inputs of the training data
    targets,        # the outputs of the training data
    epochs,         # the number of epochs to train over
    learning_rate,
    momentum = 0.0,
    batchsize = -1, # -1 means (full) batch gradient descent
    spew_init = 7,  # number of loss lines to display initially when training
    spew_end = 17,  # number of loss lines to display lastly when training
    verbosity = 3   # verbosity 0 means no printing
):
  '''
  Return the model trained with the given hyper-parameters (for epochs epochs).
  '''

  assert len(features) == len(targets),\
      "Number of features must equal number of targets."
  catch_sigint()
  loss_len = 20
  num_examples = len(features)
  if batchsize == -1: batchsize = num_examples

  if momentum != 0:
    z_params = []
    for param in model.parameters():
      z_params.append(param.data.clone())
    for param in z_params:
      param.zero_()

  if verbosity > 2:
    print(model)

  for epoch in range(epochs):

    accum_loss = 0
    indices = torch.randperm(len(features)).to(device)

    for idx in range(0, num_examples, batchsize):

      current_indices = indices[idx: idx + batchsize]
      loss = criterion(
          model(features.index_select(0, current_indices)),
          targets.index_select(0, current_indices)
      )

      accum_loss += loss.item()

      model.zero_grad()
      loss.backward()

      if momentum != 0:
        for i, (z_param, param) in enumerate(zip(z_params, model.parameters())):
          z_params[i] = momentum * z_param + param.grad.data
          param.data.sub_(z_params[i] * learning_rate)
      else:
        for param in model.parameters():
          param.data.sub_(learning_rate * param.grad.data)

    if verbosity > 0:
      base_str = "epoch {0}/{1}; loss ".format(epoch+1, epochs)
      loss_str = "{0:<10g}".format(accum_loss*batchsize/num_examples)
      if epochs < 20 or epoch < spew_init:
        print(base_str + loss_str)
      elif epoch > epochs - spew_end:
        print(end='\b'*len(base_str))
        print(base_str + loss_str)
      elif epoch == spew_init:
        print("...")
      else:
        print(' '*loss_len, end='\b'*loss_len)
        print(end='\b'*len(base_str))
        loss_len = len(loss_str)
        print(base_str+loss_str, end='\b'*loss_len, flush=True)

  if verbosity > 1:
    if learning_rate < .005:
      learning_rate_str = "learning rate: {:.4g}".format(learning_rate)
    else:
      learning_rate_str = "learning rate: {:.5g}".format(learning_rate)
    if momentum < .005:
      momentum_str = " momentum: {:.4g}".format(momentum)
    else:
      momentum_str = " momentum: {:.5g}".format(momentum)
    print(learning_rate_str, momentum_str, " batchsize:", batchsize)

  return model

def cross_validate_train(
    validation_crit=None,
    k = 10,           # the number of folds
    cent_feats=True,  # whether or not to mean-center the features
    norm_feats=True,  # whether or not to normalize the feature
    cent_targs=True,  # whether or not to mean-center the targets
    norm_targs=True,  # whether or not to normalize the targets
    model = None,     # the model to be trained
    criterion=None,   # the criterion to be used during training
    features=None,    # the inputs of the training data
    targets=None,     # the outputs of the training data
    epochs = 1,       # the number of epochs to train over for each validation
    learning_rate=None,
    momentum = 0.0,
    batchsize = -1,   # -1 means batch gradient descent
    spew_init = 7,    # number of loss lines to display initially when training
    spew_end = 17,    # number of loss lines to display lastly when training
    verbosity = 1     # verbosity 0 means no printing
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
        criterion = criterion,
        features = xss_train,
        targets = yss_train,
        epochs = epochs,
        learning_rate = learning_rate,
        momentum = momentum,
        batchsize = batchsize,
        verbosity = verbosity - 1
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
    verbosity = 1     # verbosity 0 means no printing
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
        verbosity = verbosity
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
    features,
    with_momentum = True,
    warn = True,
    sparse = False
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
):
  '''
  Compute the confusion matrix with respect to given yhatss_prob and targets.

  Note: the yhats_prob argument should be roughly model(xss), i.e., (a mini-
  batch of) probability distributions representing the prediction of a model.
  The argument yss holds the corresponding correct values.

  The columns in the returned tensor or in the diplayed table correspond
  to the actual (correct) target class and the rows are the class predicted
  by model.

  Returns total ratio (a number between 0 and 1) correctly by the model or
  optional one minus that ratio; i.e., the error rate.

  Optionally prints the confusion matrix with percentages.
  '''
  assert len(yhatss_prob) == len(yss),\
      "Number of features must equal number of targets."
  assert classes.dim() == 1,\
      "The classes argument should be a 1-dimensional tensor."

  cm_counts = torch.zeros(len(classes), len(classes))

  for yhats_prob, ys in zip(yhatss_prob, yss):
    cm_counts[torch.argmax(yhats_prob).item(), ys] += 1

  cm_pcts = cm_counts/len(yss)

  if show:
    cell_length = 6
    #fix this formatting and below, so generalizes to other len(classes)
    print(((cell_length*len(classes))//2+1)*' '+"Actual")
    print('     ',end='')
    for class_ in classes:
      print('{:{width}}'.format(class_.item(), width=cell_length),end='')
    print("\n     ------------------------------------------------------------")
    for i, row in enumerate(cm_pcts):
      print(' ',i,end=' |')
      for entry in row:
        if entry == 0.0:
          print((cell_length-1)*' '+'0', end='')
        else:
          print('{:{width}.1f}'.format(100*entry, width=cell_length), end='')
      print()

  if return_error:
    return 1-torch.trace(cm_pcts).item()
  else:
    return torch.trace(cm_pcts).item()

def r_squared(yhatss, yss, return_error = False ):
  '''
  Returns the the coefficient of determination of two 2d tensors (where the
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

def catch_sigint():
  import signal
  def keyboardInterruptHandler(signal, frame):
    #print("KeyboardInterrupt (ID: {}) caught. Cleaning up...".format(signal))
    print("\n")
    exit(0)
  signal.signal(signal.SIGINT, keyboardInterruptHandler)
