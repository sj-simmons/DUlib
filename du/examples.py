#!/usr/bin/env python3
'''that demonstrate the use and functionality of DUlib.

A note on building models:

The first argument of the `train` function in the `du.lib`
module is `model`. We assume that `model` is an instance of a
class derived from `torch.nn.Module`, as is common when work-
ing with PyTorch. Such a derived class must implement a
`forward` method (said differently, the `forward` method in
PyTorch's `nn.Module` is a virtual method). See the definition
of LinRegModel below for a simple but instructive example.

We denote by `xss` the tensor that holds the features of our
data; i.e., `xss` is the tensor that is to be forwarded by the
`forward` method of our model. We assume that `xss` is at least
2-dimensional, and that its first dimension indexes the exam-
ples of our data. For instance, suppose that we want to model a
2-dimensional regression plane that captures a point cloud that
lives in 3-space. Then `xss` is assumed to be a tensor of size
`torch.Size([n, 2])`, where `n` is the number of examples.

If `yss` denotes the corresponding targets then, even though
each example's target consists of only 1 number, we assume that
`yss` is of `torch.Size([n, 1])`.  Therefore, you might want to
employ the PyTorch utility `unsqueeze` when writing a program
that calls, say, the `train` function in this library. (See be-
low for a basic example).

                   _____________________


On training recurrent versus feedforward models:

The training functions in DUlib can be used without modifica-
tion to train both feedforward and recurrent models.  The
difference is that the features of data suitable for recurrent
nets often have variable length. So when training a recurrent
net, one often passes a tensor consisting of the lengths of the
features along with the features themselves. See the document-
ation for the various training functions for the details.

                   _____________________


The following three demonstrations of basic usage of the func-
tions in DUlib in the case of the simplest neural net: the so-
called linear perceptron.

Linear Regression:

  First, we generate some data.

  >>> import torch
  >>> xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 10.0)

  The x-values above are selected uniformly from the interval
  [0, 100].  The y-values were obtained by adding normally dis-
  tributed error to y=2x+9 when x runs through the xs.

  Let us next cast the data as tensors of size appropriate for
  training a neural net.

  >>> xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)
  >>> xss.size(); yss.size()
  torch.Size([40, 1])
  torch.Size([40, 1])

  For best performance, we center and normalize the data.

  >>> from du.lib import center, normalize
  >>> xss,xss_means = center(xss); yss,yss_means = center(yss)
  >>> xss,xss_stds=normalize(xss); yss,yss_stds=normalize(yss)

  Next, let us create an instance a model that computes the
  linear regression line (which should be close to y=2x+9).

  >>> import torch.nn as nn
  >>> class LinRegModel(nn.Module):
  ...   def __init__(self):
  ...     super(LinRegModel, self).__init__()
  ...     self.layer = nn.Linear(1, 1)
  ...   def forward(self, xss):
  ...     return self.layer(xss)
  >>> model = LinRegModel()

  We now specify a loss function, compute the optimal learning
  rate and momentum, and train our model.

  >>> criterion = nn.MSELoss()
  >>> from du.lib import train
  >>> model = train(
  ...     model = model,
  ...     crit = criterion,
  ...     train_data = (xss, yss),
  ...     eps = 50,
  ...     verb = 0)

  Suppose that we want to predict the y-value when the x-value
  is 50 (this should be close to 2*50+9).

  >>> testss = torch.tensor([50.]).unsqueeze(1)
  >>> testss; testss.size()
  tensor([[50.]])
  torch.Size([1, 1])

  We mean center and normalize with respect to the means and
  standard deviations of the training data.

  >>> testss, _ = center(testss, xss_means)
  >>> testss, _ = normalize(testss, xss_stds)

  After running the inputs for which we wish to make an predic-
  tion through our trained model, we translate the output to
  where it is supposed to be.

  >>> yhatss = model(testss)
  >>> prediction = (yhatss.mul_(yss_stds)+yss_means).item()
  >>> abs(prediction - 109) < 5
  True

Linear Regression with learning rate decay:

  The data, which are already centered and normalized are
  those of the previous example. First we re-instance the
  model, thereby re-initialing the weights.

  >>> model = LinRegModel()

  Let us implement a dynamic learning rate that decays over
  time.

  >>> learning_rate = 0.1; epochs = 2000
  >>> decay_rate = 1-75*learning_rate/epochs
  >>> print(decay_rate)
  0.99625
  >>> adaptives = {'lr': lambda x: decay_rate * x}

  And train the model.

  >>> model = train(model, criterion, (xss, yss), eps = epochs,
  ...     lr = learning_rate, adapts = adaptives, verb = 0)

  Now we check that the weights of our model converged to about
  2 and 9, the slope and intercept of the line we used to gen-
  erate the original data.

  >>> params = list(model.parameters())
  >>> mm = params[0].item(); bb = params[1].item()

  Now map the weights back to unnormalized/uncentered data, and
  check that the slope and intercept are close to 2 and 9,

  >>> my=yss_means.item(); mx=xss_means.item()
  >>> sy=yss_stds.item(); sx=xss_stds.item()
  >>> slope = mm*sy/sx; intercept = my+bb*sy-slope*mx
  >>> all([abs(slope - 2)  < 0.1, abs(intercept - 9.0) < 6.0])
  True

Linear Regression without normalizing or centering:

  There is no reason not to center and normalize for this
  problem. But, just for the sport of it, one can use the
  `optimize_ols` function:

  >>> xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 10.0)
  >>> xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)
  >>> from du.lib import optimize_ols
  >>> lr, mo = optimize_ols(xss, verb = 0)
  >>> model = train(model, criterion, (xss, yss), lr = lr,
  ...     mo = mo, eps = 3000, verb = 0)
  >>> params = list(model.parameters())
  >>> slope = params[0].item(); intercept = params[1].item()
  >>> all([abs(slope - 2)  < 0.1, abs(intercept - 9.0) < 6.0])
  True

                   _____________________


  Entire programs that employ the complete functionality of
  DUlib can be found at the DL@DU Project.

                   _____________________

'''
__author__ = 'Simmons'
__version__ = '0.6'
__status__ = 'Development'
__date__ = '11/17/19'

if __name__ == '__main__':
  import doctest
  doctest.testmod()

