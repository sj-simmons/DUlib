#!/usr/bin/env python3
'''that demonstrate the functionality of `DUlib`.

Neural nets provide a way to learn from data. The weights of a
well-trained neural net are a reflection, or ~representation~,
of the data on which it was trained. The goal of Deep Learning
is to find useful representation of the data in which we are
interested.

In addition to finding training parameters that lead to conver-
gence under training,  we must design with well-conceived arch-
itecture of the neural net itself. Otherwise, the weights have
little, if any, chance of revealing a useful representation of
the data, whether or not the model converges.

_Notes on training data_

The essential function provided by DUlib is `train`; which we
use in our programs by first importing it with, for example:

    `from du.lib import train`

The first argument of the `train` function is `model`. We as-
sume that `model` is an instance of a class derived from the
PyTorch class `torch.nn.Module`. Such a derived class must
implement a `forward` method (that is, the `forward` method
of `nn.Module` class is a virtual method).

(See the definition of `LinRegModel` below for a simple but
instructive example of sub-classing `nn.Module`, creating an
instance of that subclass, and using that instance to train
a model to solve a simple linear regression problem.)

Throughout DUlib, we denote by `xss` the tensor that holds the
~features~ (or ~inputs~) of our data; i.e., `xss` is the tensor
that is to be forwarded by the `forward` method of our model.
We assume that `xss` is at least 2-dimensional, and that its
first dimension indexes the examples of our data.

For instance, suppose that we want to model a 2-dimensional
regression plane that captures a point cloud which lives in
3-space. In this case, `xss` is assumed to be a tensor of size
`torch.Size([n, 2])`, where `n` is the number of examples.

As is the convention in PyTorch's documentation, let us agree
to write `(n,2)` instead of `torch.Size([n, 2])`, for example,
and refer to the 'shape' of a tensor, though we use 'shape' and
'size' interchangeably.

Later, when we want to classify or otherwise model images (all
of which we will assume to be the same size), our features will
have shape `(n, image_width, image_height)` where `n` is the
number of images. Let us agree to denote this as `(n,*)` where,
in fact, there could be any number of additional dimensions be-
yond the first.

As stated above, we assume that our features `xss` are always
at least 2-dimensional; said differently, we assume that `xss`
has shape `(n,*)` where `n` is the number of features. This is
true even in the simplest case in which the features of an ex-
ample in our data consists of a single number. In that case,
`xss` should be a tensor of shape (n,1), not (n).

(Therefore, you may wish, in your code, to deploy the PyTorch
utility `unsqueeze` when writing a program that calls, say, the
`train` function in this library.  Again, see below for a basic
example.)

Now, given features denoted `xss` we, throughout, denote the
corresponding targets (or outputs) by `yss`. What is the conv-
ention surrounding the shape of the targets `yss`?

Suppose the scenario mentioned above: that of modeling a point
cloud living in R^3 with a 2-d plane. Then, our `xss` tensor
would naturally have shape `(n,2)` where, yet again, `n` denotes
the number of examples (i.e., the number of points in our
point cloud). What shape should `yss` be in this scenario?

In this case, the machinery in DUlib would assume that the cor-
responding `yss` have shape `(n,1)`.  This may seem unnatural.
Should not, more simply, the `yss` have shape `(n)` since each
example's target consists of a single number?

The reason that we prefer `yss` to be of shape `(n,1)` instead
of just `(n)` is that, in general, the examples' targets can
consist any number of numbers. We can easily imagine scenarios
in which we want to model a function with `k` inputs and `m`
outputs. Then `xss` would be of shape `(n,k)` while `yss` would
have shape `(n,m)` (and there would be `n` examples).

As we stated above, `xss` should always be of shape `(n,*)`
(meaning it should be at least 2 dimensional).  It may seems
reasonable to impose the same convention for targets `yss`.
And, for, say, regression problems, yes, DUlib assumes that
the `yss` have shape `(n,*)`. But there is an exception that
occurs very commonly.

In classification problems (in which, for example, say we want
to classify images as landscapes, city-scapes, or see- scapes),
then the target for each image would naturally and simply be an
int: either `0`, `1`, or `2`.

Furthermore, it makes no sense, in commonly held practice to
try to map an image to say both a sea-scape or a land-scape.
Rather, if we wanted something like that we would, after train-
ing, easily use the model to get, for given set of features,
our hands on the entire discrete probability distribution of
the trained model's best guesses over all of the target classes.

In summary, the `xss` are always assumed to be at least 2-dimen-
sional, and so are the `yss`, unless we are working on a class-
ification problem, in which case the `yss` are assumed to be
one dimensional. So that, in the case of a classification pro-
blem, if `xss` has shape `(n,*)`, then `yss` would have shape
simply `(n)`.

Lasty, each entry in `yss` should be an `int` in the range `0`
to `C-1` where `C` is the number of classes. Importantly, `yss`
must be a `torch.LongTensor` for a classification problem.

A final note on notation: we use `yhatss` thoughout to denote
the predictions made by a trained model on features unseen
during training.
                    _____________________

The following are three demonstrations of basic usage of the
functionality of DUlib in the case of the simplest neural net:
the so called linear perceptron.

_Simple linear regression_

  First, we generate some data.

  >>> $import torch$
  >>> $xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 10.0)$

  The `x`-values above are selected uniformly from the interval
  `[0, 100]`.  The `y`-values were obtained by adding normally
  distributed error to `y=2x+9` when `x` runs through the `xs`.

  Let us next cast the data as tensors of size appropriate for
  training a neural net.

  >>> $xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)$
  >>> $xss.size(); yss.size()$
  |torch.Size([40, 1])|
  |torch.Size([40, 1])|

  For best performance, we center and normalize the data.

  >>> $from du.lib import center, normalize$
  >>> $xss,xss_means = center(xss); yss,yss_means = center(yss)$
  >>> $xss,xss_stds=normalize(xss); yss,yss_stds=normalize(yss)$

  Next, let us create an instance a model that computes the
  ~least-squares regression~ line (which should be close to
  `y=2x+9`).

  >>> $import torch.nn as nn$
  >>> $class LinRegModel(nn.Module):$
  ...   $def __init__(self):$
  ...     $super().__init__()$
  ...     $self.layer = nn.Linear(1, 1)$
  ...   $def forward(self, xss):$
  ...     $return self.layer(xss)$
  >>> $model = LinRegModel()$

  We now specify a ~loss function~, compute the optimal ~learning~
  ~rate~ and ~momentum~, and train our model.

  >>> `criterion = nn.MSELoss()`
  >>> `from du.lib import train`
  >>> `model = train(`
  ...     `model = model,`
  ...     `crit = criterion,`
  ...     `train_data = (xss, yss),`
  ...     `learn_params = {'lr': 0.1},`
  ...     `epochs = 50,`
  ...     `verb = 0)`

  Suppose that we want to predict the `y`-value associated to the
  `x`-value `50`. If `50` happens to be `x`-value in the data set, we
  could just take for the prediction the corresponding `y`-value
  (or the average of the corresponding `y`-values if `50` happens to
  occur more than once.

  If `50` does not occur, we could use the regression line to make
  our prediction (this should be close to `2*50+9`). But notice that,
  even if `50` does occur, we still probably want to use the regres-
  sion line, since we are assuming that the original data includes
  error.

  >>> `testss = torch.tensor([50.]).unsqueeze(1)`
  >>> `testss; testss.size()`
  `tensor([[50.]])`
  `torch.Size([1, 1])`

  We mean `center` and `normalize` with respect to the means and
  standard deviations of the training data.

  >>> `testss, _ = center(testss, xss_means)`
  >>> `testss, _ = normalize(testss, xss_stds)`

  After running the inputs for which we wish to make an predic-
  tion through our trained model, we translate the output to
  where it is supposed to be.

  >>> `yhatss = model(testss)`
  >>> `prediction = (yhatss.mul_(yss_stds)+yss_means).item()`
  >>> `abs(prediction - 109) < 5`
  `True`

_Simple linear regression with learning rate decay_

  The data, which are already centered and normalized are
  those of the previous example. First we re-instance the
  model, thereby re-initialing the weights. The criterion
  is still `MSELoss`.

  >>> model = LinRegModel()

  Let us use the class LearnParam_ to implement a dynamic
  learning rate that decays over time.

  >>> from du.lib import LearnParams_
  >>> class LR_decay(LearnParams_):
  ...   def __init__(self, lr, rate):
  ...     super().__init__(lr)
  ...     self.rate = rate
  ...   def update(self, params):
  ...     self.lr = self.rate * self.lr
  ...     super().update(params)

  Now we train using an instance of the above class.
  >>> learning_rate = 0.1; epochs = 2000
  >>> decay_rate = 1-75*learning_rate/epochs
  >>> print(decay_rate)
  0.99625
  >>> model = train(
  ...   model,
  ...   criterion,
  ...   (xss, yss),
  ...   learn_params = LR_decay(learning_rate, decay_rate),
  ...   epochs = epochs,
  ...   verb = 0)

  Now we check that the weights of our model converged to about
  2 and 9, the slope and intercept of the line we used to gen-
  erate the original data.

  >>> params = list(model.parameters())
  >>> m = params[0].item(); b = params[1].item()

  Now map the weights back to unnormalized/uncentered data, and
  check that the slope and intercept are close to 2 and 9,

  >>> my=yss_means.item(); mx=xss_means.item()
  >>> sy=yss_stds.item(); sx=xss_stds.item()
  >>> slope = m*sy/sx; intercept = my+b*sy-slope*mx
  >>> all([abs(slope - 2)  < 0.1, abs(intercept - 9.0) < 10.0])
  True

Simple linear regression without normalizing or centering:

  There is no reason not to center and normalize for this
  problem. But, just for the sport of it, one can use the
  `optimize_ols` function:

  >>> model = LinRegModel()
  >>> xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 10.0)
  >>> xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)
  >>> from du.lib import optimize_ols
  >>> model = train(
  ...     model = model,
  ...     crit = criterion,
  ...     train_data = (xss, yss),
  ...     learn_params = optimize_ols(xss),
  ...     epochs = 3000,
  ...     verb = 0)
  >>> params = list(model.parameters())
  >>> slope = params[0].item(); intercept = params[1].item()
  >>> all([abs(slope - 2)  < 0.1, abs(intercept - 9.0) < 10.0])
  True

  Another way to validate the models above is simply to compute
  r^2, which is called the coefficient of determination. For the
  last example, r^2 can be computed by

  >>> from du.lib import r_squared
  >>> yhatss = model(xss)
  >>> r_squared(yhatss, yss) # doctest: +SKIP
  .9711...

  This means that about 97% of the variation in the data is ex-
  plained by the regression line. Said differently, the model
  performs very well.
                    _____________________

_Note on visualizations_

If you have `matplotlib` installed, you can easily take in some
visualizations of the ideas in the last sections.

To see the graph of the 40-point point-cloud along with the or-
iginal line (y=2x+9) that we used to generate the cloud, along
with the regression line that we found using gradient descent
(via our code in the last demo), simply type this at the com-
mand line:

  dulib_linreg

Run the program a couple of times. Each time, the initial point
cloud is slightly different since we add noise when we generate
it.

You can watch the models best guess as to the best fit regres-
sion line improve over each epoch of training by type at the
command line

  dulib_linreg_anim
                    _____________________

Polynomial regression

  The three demonstration above were examples of simple linear
  regression.  More generally, (ols) polynomial regression ref-
  ers to fitting a polynomial (of pre-specified degree) to data
  in a way optimimal in the least-squares sense.

  In this example both our features and targets will have shape
  `(n,1)`.  To fit higher degree polynomials (so quadratic,
  cubics, etc. rather than just lines), we regress over not
  just the xs in xss but also over xs^2, xs^3, ect. (we get
  constant term in our polynomial since our model classes have
  a bias by default).

  Let us generate some data by adding noise to a sampled non-
  linear function.

  >>> xs = 100*torch.rand(40)
  >>> ys = torch.normal(20*torch.sin(xs)+9, 10.0)
  >>> xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  And try to fit a polynomial of the following degree.

  >>> degree = 5

  We need to build a new model class.

  >>> class SimpleLinReg(nn.Module):
  ...   def __init__(self, degree):
  ...     super().__init__()
  ...     self.deg = degree
  ...     self.layer = nn.Linear(degree, 1)
  ...   def forward(self, xss):
  ...     #copy xss to the cols of a (len(xss),deg) tensor
  ...     powers = xss * torch.ones(len(xss), self.deg)
  ...     #square entries in 2rd col, cube those in the 3rd,...
  ...     powers = powers.pow(torch.arange(1.,self.deg+1))
  ...     return self.layer(powers)

  Let us instance and train this new class.

  >>> model = train(
  ...     model = SimpleLinReg(degree),
  ...     crit = nn.MSELoss(),
  ...     train_data = (xss, yss),
  ...     learn_params = {'lr': 0.00001, 'mo': 0.9},
  ...     epochs = 3000,
  ...     verb = 0)

  Let's check how well this trained model works.

  >>> test_xss=(100*torch.rand(40)).unsqueeze(1)
  >>> test_ys = torch.normal(20*torch.sin(xs)+9, 10.0)
  >>> test_yss = test_ys.unsqueeze(1)
  >>> yhatss = model(test_xss)
  >>> from du.lib import r_squared
  >>> r_squared(yhatss, test_yss)

                    _____________________

Nonlinear regression

  >>> xs = 100*torch.rand(40)
  >>> ys = torch.normal(20*torch.sin(xs)+9, 10.0)
  >>> xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)
  >>> from du.models import OneLayerFC
  >>> #model = OneLayerFC(width = 10)

                    _____________________

Entire programs that employ the complete functionality of DUlib
can be found at The DL@DU Project.

'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from du.lib import optimize_ols, train
from du.util import _markup

__author__ = 'Scott Simmons'
__version__ = '0.8.5'
__status__ = 'Development'
__date__ = '12/06/19'
__doc__ = _markup(__doc__)

class LinRegModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer = nn.Linear(1, 1)
  def forward(self, xss):
    return self.layer(xss)

def simple_linear_regression():
  import argparse

  xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 20.0)
  xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  d = optimize_ols(xss)
  parser = argparse.ArgumentParser(
      description = 'Simple linear regression via gradient descent',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-lr',type=float,help='learning rate',default=d['lr'])
  parser.add_argument('-mo',type=float,help='momentum',default=d['mo'])
  parser.add_argument('-epochs',type=int,help='epochs',default=3000)
  parser.add_argument('-gr',type=int,help='1 to show loss',default=0)
  args = parser.parse_args()

  model = LinRegModel()
  model = train(
      model = model,
      crit = nn.MSELoss(),
      train_data = (xss, yss),
      learn_params = {'lr':args.lr, 'mo':args.mo},
      epochs = args.epochs,
      graph = args.gr,
      verb = 2)
  params = list(model.parameters())
  slope = params[0].item(); intercept = params[1].item()

  fig, _ = plt.subplots()
  plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
  plt.scatter(xs.tolist(),ys.tolist(),s=9)
  xs = torch.arange(101.)
  plt.plot(xs, 2*xs+9, c='black', lw=.5, label='y = 2x + 9')
  plt.plot(xs, slope*xs+intercept, c='red', lw=.9,\
      label='reg. line: y = {:.2f}x + {:.2f}'.format(slope, intercept))
  plt.legend(loc=1);
  plt.show()

def simple_linear_regression_animate():
  import argparse
  from du.lib import train, optimize_ols

  xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 50.0)
  xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  d = optimize_ols(xss)
  parser = argparse.ArgumentParser(
      description = 'Simple linear regression via gradient descent',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-lr',type=float,help='learning rate',default=d['lr'])
  parser.add_argument('-mo',type=float,help='momentum',default=d['mo'])
  parser.add_argument('-epochs',type=int,help='epochs',default=3000)
  parser.add_argument('-bs',type=int,help='batchsize <=40',default=-1)
  args = parser.parse_args()

  plt.ion(); fig, _ = plt.subplots()
  plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
  xs_ = torch.arange(101.)
  plt.plot(xs_, 2*xs_+9, c='black', lw=.5, label='y = 2x + 9')
  plt.scatter(xs.tolist(),ys.tolist(),s=9)

  model = LinRegModel()
  for epoch in range(args.epochs):
    model = train(
        model = model,
        crit = nn.MSELoss(),
        train_data = (xss, yss),
        learn_params = {'lr':args.lr, 'mo':args.mo},
        epochs = 1,
        bs = args.bs,
        verb = 0)
    params = list(model.parameters())
    slope = params[0].item(); intercept = params[1].item()

    plt.clf()
    plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
    plt.scatter(xs.tolist(),ys.tolist(),s=9)
    plt.plot(xs_, 2*xs_+9, c='black', lw=.5, label='y = 2x + 9')
    plt.plot(xs_, slope*xs_+intercept, c='red', lw=.9,\
      label='reg. line: y = {:.2f}x + {:.2f}'.format(slope, intercept))
    plt.legend(loc=1)
    try:
      fig.canvas.flush_events()
    except tkinter.TclError:
      plt.ioff()
      exit()
  plt.ioff()
  plt.show()

def simple_polynomial_regression():
  import argparse
  from du.models import polyize, SimpleLinReg
  from du.lib import optimize_ols

  num_points = 20; x_width = 40.0; h_scale = 1.5; v_shift = 5
  xs = x_width*torch.rand(num_points) - x_width /h_scale
  ys = torch.normal(2*xs*torch.cos(xs/10)-v_shift, 10.0)
  xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  parser = argparse.ArgumentParser(
      description =\
         'Simple poly regression via gradient descent'+
         '\n  put lr = -1 to try optimal learning params',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-deg',type=int,help='degree of poly',default=3)
  parser.add_argument('-lr',type=float,help='learning rate',default=1e-9)
  parser.add_argument('-mo',type=float,help='momentum',default=.999)
  parser.add_argument('-epochs',type=int,help='epochs',default=10000)
  parser.add_argument('-gr',type=int,help='1 to show loss',default=0)
  parser.add_argument('-show_opt',\
      help='show optimal learning parameters and quit',action='store_true')
  args = parser.parse_args()

  degree = args.deg
  xss = polyize(xss, degree)
  print('degree is',degree)

  if args.show_opt:
    print(optimize_ols(xss, verb=2))
    exit()
  if args.lr < 0 or args.mo < 0:
    learn_params = optimize_ols(xss, verb=1)
  else:
    learn_params = {'lr':args.lr, 'mo':args.mo}

  model = train(
      model = SimpleLinReg(degree),
      crit = nn.MSELoss(),
      train_data = (xss, yss),
      learn_params = learn_params,
      epochs = args.epochs,
      graph = args.gr,
      verb = 2)

  fig, _ = plt.subplots()
  plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
  plt.scatter(xs.tolist(),ys.tolist(),s=9,
      label='y = 2x*cos(x/10)-{}+10*N(0,1)'.format(v_shift))
  xs_ = torch.arange(float(int(x_width)+1)) - x_width/h_scale;
  plt.plot(xs_, xs_*torch.cos(xs_/10)-v_shift, c='black', lw=.5,\
      label='y = 2x*cos(x/10)-{}'.format(v_shift))
  yhatss = model(polyize(xs_.unsqueeze(1),degree)).squeeze(1)
  plt.plot(xs_, yhatss.detach(), c='red', lw=.9,
      label='reg. poly (deg={})'.format(degree))
  plt.legend(loc=1);
  plt.show()

  #test_xss=(100*torch.rand(40)).unsqueeze(1)
  #test_ys = torch.normal(20*torch.sin(xs)+9, 10.0)
  #test_yss = test_ys.unsqueeze(1)
  #yhatss = model(polyize(test_xss, degree))
  #from du.lib import r_squared
  #r_squared(yhatss, test_yss)

if __name__ == '__main__':
  simple_polynomial_regression()
  import doctest
  doctest.testmod()
