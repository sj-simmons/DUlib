#!/usr/bin/env python3
'''that demonstrate the functionality of `du.lib`.

~Neural nets~ provide a way to learn from data. The ~weights~ of a
well-trained neural net are a reflection, or ~representation~, of
the data on which it was trained. The objective in Machine and
Deep Learning is to find representations which yield patterns
in the data at hand.

In addition to finding parameters that lead to convergence un-
der training, we must design with well-conceived architecture
of the neural net itself. Otherwise, the weights have little
chance of revealing a useful representation of the data, wheth-
er or not the model converges.

!Notes on training data!

The essential function provided by DUlib is `train`; which we use
in our programs by first importing it with, for example:

  `from du.lib import train`

The first argument of the `train` function is `model`. (Recall that
`pd du.lib.train` quickly returns, at your command line, the us-
age of `train`.) We assume that `model` is an instance of a class
derived from the `PyTorch` class `torch.nn.Module`. Such a derived
class must implement a `forward` method; that is, the `forward` me-
thod of the `nn.Module` class is a virtual method.

(See the definition of `LinRegModel` below for a simple but inst-
ructive example that sub-classes `nn.Module` and then creates an
instance of that subclass and uses it to train a model to solve
a simple linear regression problem.)

We often denote by `xss` the tensor that holds the ~features~ (or
~inputs~) of our data; i.e., `xss` is the tensor that is to be for-
warded by the `forward` method of our model. We assume that `xss`
is of dimension at least 2, and that its first dimension index-
es the examples of our data.

For instance, suppose that we want to model a 2-dimensional re-
gression plane that captures a point cloud which lives in R^3.
In this case, `xss` is assumed to have size `torch.Size([n, 2])`,
where `n` is the number of examples (points).

As is the convention in PyTorch's documentation, let us agree
to write `(n,2)` instead of `torch.Size([n, 2])`, for example, and
refer to the 'shape' of a tensor instead of its 'size', though
we use 'shape' and 'size' interchangeably.

Later, when we want to classify or otherwise model images (all
of which we will assume to be the same size), our features will
have shape `(n, image_height, image_width)` where `n` is the number
of images. Let us agree to denote this as `(n,*)` where, in fact,
there could be any number of additional dimensions beyond the
first.

As stated above, we assume that our features `xss` are always at
least 2-dimensional; stated differently, we assume that `xss` has
shape `(n,*)`; and, moreover, that `n` is the number of examples in
our data. This is true even in the simplest case: that in which
the features of an example in our data consists of a single nu-
mber. In that case, `xss` should of shape `(n,1)`, not `(n)`.

(Therefore, you may wish, in your code, to deploy the `PyTorch`
utility `unsqueeze` when writing a program that calls, say, the
`train` function in this library. Again, see below for basic exa-
mples.)

Given features (or inputs) denoted `xss` we, throughout, notate
the corresponding ~targets~ (or ~outputs~) by `yss`. What is the con-
vention surrounding the shape of the targets `yss`?

Suppose the scenario mentioned above: that of modeling a point
cloud living in R^3 with a 2-dimensional plane. Then, our `xss`
tensor would naturally have shape `(n,2)` where, yet again, `n` de-
notes the number of examples (i.e., the number of points in our
point cloud). What shape should `yss` be in this scenario?

In this case, the machinery of `DUlib` would assume that the cor-
responding `yss` has shape `(n,1)`. This may seem unnatural. Should
not, more simply, the `yss` have shape `(n)` since each example's
target consists of a single number?

The reason that we prefer `yss` to be of shape `(n,1)` instead of
just `(n)` is that, in general, the examples' targets can consist
any number of numbers. We can easily imagine scenarios in which
we want to model a function with `k` inputs and `m` outputs. Then
`xss` would be of shape `(n,k)` while `yss` would have shape `(n,m)`
(and there would be `n` examples in our data set).

As we stated above, `xss` should always be of shape `(n,*)` (mean-
ing it should be at least 2 dimensional). It may seem reasonab-
le to impose the same convention for targets `yss`. And for, say,
regression problems, yes, `DUlib` assumes that the `yss` have shape
`(n,*)`. But there is an exception that occurs often.

In classification problems (in which, for instance, we want to
classify images as say landscapes, city-scapes, or see-scapes),
the target for each image would naturally and simply be an `int`:
either `0`, `1`, or `2`.

Furthermore, it makes little sense, in commonly held practice,
to try to map an image to say both a sea-scape or a land-scape.
Rather, if we wanted something like that we would, after train-
ing, easily use the model to get, for a given set of features,
our hands on the entire ~discrete probability distribution~ of
the trained model's best guesses over all target classes.

In summary, the `xss` are always assumed to be at least 2 dimen-
sional, and so are the `yss`, unless we are working on a classif-
ication problem, in which case the `yss` are assumed to be one
dimensional; so that, in the case of a classification problem,
if `xss` has shape `(n,*)`, then `yss` would have shape simply `(n)`.
In any case, the first dimension of a data tensor indexes the
examples in (a mini-batch of) our data.

Lastly, for a classification problem, each entry in `yss` should
be an `int` in the range `0` to `C-1` where `C` is the number of class-
es. Importantly, `yss` must be a `torch.LongTensor` in the case of
a classification problem.

Final notes on notation:

 - We often use `yhatss` to denote predictions made using a model
   on features unseen during training.

 - When writing a program that solves a particular problem (so
   that we have an actual data set that we are working with),
   we often use `feats` and `targs` for the inputs, respectively,
   outputs.

   When writing code, such as libraries in DUlib, applicable to
   general problems and their data sets, we often use `xss` resp.
   `yss`, as described above.

                    _____________________


The following demonstrations illustrate the core functionality
of `DUlib` using the case of the simplest neural net, the so cal-
led ~linear perceptron~.

!Simple linear regression!

First, we generate some data.

>>> `import torch`
>>> `xss = 100*torch.rand(40)`
>>> `yss = torch.normal(2*xss+9, 10.0)`

The `x`-values, `xss`, above are selected uniformly from the inter-
val `[0, 100]`. The `y`-values were obtained by adding normally di-
stributed error to `y=2x+9` as `x` runs through the `xs`.

Let us next cast the data as tensors of a size appropriate for
training a neural net using the `train` function in `DUlib`. (The
`train` function is in the core library `du.lib`; so you can per-
use the documentation for the `train` function by typing, for ex-
ample, `pd du.lib.train` at the command line.)

>>> `feats = xss.unsqueeze(1); targs = yss.unsqueeze(1)`
>>> `feats.size(); targs.size()`
$torch.Size([40, 1])$
$torch.Size([40, 1])$

For best performance, we ~center~ and ~normalize~ the data, saving
for later the mean and standard deviation of each of the inputs
and the outputs.

>>> `from du.lib import center, normalize`
>>> `feats, feats_mean = center(feats)`
>>> `feats, feats_std = normalize(feats)`
>>> `targs, targs_mean = center(targs)`
>>> `targs, targs_std = normalize(targs)`

Next, let us create an instance of a model that, upon training,
computes the ~least-squares regression~ line for the given data.
(The regression line should be close to the line `y=2x+9`).

>>> `import torch.nn as nn`
>>> `class LinRegModel(nn.Module):`
...   `def __init__(self):`
...     `super().__init__()`
...     `self.layer = nn.Linear(1, 1)`
...   `def forward(self, xss):`
...     `return self.layer(xss)`
>>> `model = LinRegModel()`

Note: using the functionality of `DUlib`, we can quickly define
the previous class with

>>> `from du.models import DenseFFNet`
>>> `model = DenseFFNet(1, 1)`

This is equivalent to the definition above with one added bene-
fit that has to do with serialization. Type `pd du.models` in or-
der to read about this.)

We now specify a ~loss function~ and train our model.

>>> `criterion = nn.MSELoss()`
>>> `from du.lib import train`
>>> `model = train(`
...     `model = model`,
...     `crit = criterion`,
...     `train_data = (feats, targs)`,
...     `learn_params = {'lr': 0.1}`,
...     `epochs = 50`,
...     `verb = 0)`

Suppose that we want to predict the `y`-value (or output, or tar-
get) associated to the `x`-value (or input, or feature) 50. If 50
happens to be an `x`-value in the data set, we could just take
for the prediction the corresponding `y`-value (or the average of
the corresponding `y`-values if 50 happens to occur more than
once as an input of data).

If 50 does not occur, we can use the regression line to make
our prediction (which should be close to 2*50+9, by the way).
But notice that, even if 50 does occur in the data, we still
want to use the regression line since we are assuming that the
original data includes error.

>>> `xs = torch.tensor([50.]).unsqueeze(1)`
>>> `xs; xs.size()`
$tensor([[50.]])$
$torch.Size([1, 1])$

We mean center and normalize with respect to the means and
standard deviations of the training data.

>>> `xs, _ = center(xs, feats_mean)`
>>> `xs, _ = normalize(xs, feats_std)`

After running the inputs for which we wish to make a predict-
ion through our trained model, we translate the output to where
it is supposed to be.

>>> `yhats = model(xs)`
>>> `prediction = (yhats.mul_(targs_std)+targs_mean).item()`
>>> `abs(prediction - 109) < 5`
$True$

The last line above checks that the output for the input 50 is
fairly close to 109=2*50+9. Why is this not exactly 109?

!Simple linear regression with learning rate decay!

The data for this demonstration are those of the previous one,
which are already centered and normalized. First we re-instance
the model, thereby re-initialing the weights. The criterion is
still `MSELoss`.

>>> `model = DenseFFNet(1,1)`

Let us use the class `LearnParams_` in `DUlib` to implement a dyna-
mic learning rate that decays over time. (Read the documenta-
tion for `LearnParams_`, which is a convenient base class for im-
plementing various adaptive schemes that tune learning hyper-
parameters, by issuing `pd du.lib.LearnParams_` at your command
line.)

>>> `from du.lib import LearnParams_`
>>> `class LR_decay(LearnParams_):`
...   `def __init__(self, lr, rate):`
...     `super().__init__(lr)`
...     `self.rate = rate`
...   `def update(self, params):`
...     `self.lr = self.rate * self.lr`
...     `super().update(params)`

Now we train using an instance of the above class.

>>> `learning_rate = 0.1; epochs = 2000`
>>> `decay_rate = 1-85*learning_rate/epochs`
>>> `print(decay_rate)`
$0.99575$
>>> `model = train(`
...   `model = model`,
...   `crit = criterion`,
...   `train_data = (feats, targs)`,
...   `learn_params = LR_decay(learning_rate, decay_rate)`,
...   `epochs = epochs`,
...   `verb = 0)`

Let us check that the weights of our model converged to about 2
and 9, the slope and intercept of the line we used to generate
the original data.

>>> `params = list(model.parameters())`
>>> `m = params[0].item(); b = params[1].item()`

Now map the weights back to unnormalized/uncentered data, and
check that the slope and intercept are close to 2 and 9,

>>> `my=targs_mean.item(); mx=feats_mean.item()`
>>> `sy=targs_std.item(); sx=feats_std.item()`
>>> `slope = m*sy/sx; intercept = my+b*sy-slope*mx`
>>> `all([abs(slope - 2)  < 0.2, abs(intercept - 9.0) < 13.0])`
$True$

Why are the numbers not exactly 2 and 9?

!Simple linear regression without normalizing or centering!

There is no reason not to center and normalize for this prob-
lem. But, for the sport of it, one can use the `optimize_ols`
function in `DUlib` as follows:

>>> `model = DenseFFNet(1,1)`
>>> `xss = 100*torch.rand(40); yss=torch.normal(2*xss+9, 10.0)`
>>> `feats = xss.unsqueeze(1); targs = yss.unsqueeze(1)`
>>> `from du.lib import optimize_ols`
>>> `model = train(`
...     `model = model`,
...     `crit = criterion`,
...     `train_data = (feats, targs)`,
...     `learn_params = optimize_ols(feats)`,
...     `epochs = 4000`,
...     `verb = 0`)
>>> `params = list(model.parameters())`
>>> `slope = params[0].item(); intercept = params[1].item()`
>>> `all([abs(slope - 2)  < 0.2, abs(intercept - 9.0) < 13.0])`
$True$

Another way to validate the trained model above is to compute
r^2, the ~coefficient of determination~. For the current demon-
stration, r^2 can be computed with

>>> `from du.lib import explained_var`
>>> `0.96 < explained_var(model, (feats, targs)) < 0.985`
$True$

This means that about 97% of the variation in the data is ex-
plained by the regression line; i.e., the model performs very
well. Why is this not 100%?

                    _____________________


!Note on visualizations!

If you have `matplotlib` installed, you can easily take in some
visualizations of the ideas in the last section.

To see the graph of the 40-point point-cloud along with the or-
iginal line (y=2x+9) that we used to generate the cloud along
with the regression line that we found using gradient descent,
simply type this at the command line:

  `dulib_linreg`

Run the program a couple of times. Each time, the initial point
cloud is slightly different since we add noise when we generate
it.

You can watch the models best guess as to the best fit regres-
sion line improve over each epoch of training by typing at the
command line

  `dulib_linreg_anim`

                    _____________________


!Polynomial regression!

The demonstrations above were examples of simple linear regres-
sion.  More generally, (ols) ~polynomial regression~ refers to
fitting a polynomial (of pre-specified degree) to data in a way
that is optimal in the least-squares error sense.

In this demo, both our features and targets have shape `(n,1)`.
To fit higher degree polynomials (so quadratics, cubics, etc.
rather than just lines), we regress over not just the `xs` in
`xss` but also over `xs`^2, `xs`^3, and so on. (we get the constant
term of our polynomial for free since our model classes include
a bias by default).

Let us generate some data by adding i.i.d. noise to a sampled
non-linear function:

>>> `xss = 40*torch.rand(20)-80/3`
>>> `yss = torch.normal(2*xss*torch.cos(xss/10)-5, 10.0)`
>>> `feats = xss.unsqueeze(1); targs = yss.unsqueeze(1)`

and try to fit to those data a polynomial of degree:

>>> `degree = 3`

We need to build a new model class.

>>> `class SimpleLinReg(nn.Module):`
...   `def __init__(self, degree):`
...     `super().__init__()`
...     `self.deg = degree`
...     `self.layer = nn.Linear(degree, 1)`
...   `def forward(self, xss):`
...     #copy xss to the cols of a (len(xss),deg) tensor
...     `powers = xss * torch.ones(len(xss), self.deg)`
...     #square entries in 2rd col, cube those in the 3rd,...
...     `powers = powers.pow(torch.arange(1.,self.deg+1))`
...     `return self.layer(powers)`

Let us instance and train this new class.

>>> `model = train(`
...     `model = SimpleLinReg(degree)`,
...     `crit = nn.MSELoss()`,
...     `train_data = (feats, targs)`,
...     `learn_params = {'lr': 1e-9, 'mo': 0.999}`,
...     `graph = 0,`
...     `epochs = 8000`,
...     `verb = 0)`

Let us now compute r^2 for regression polynomial found by the
model, though we should ask ourselves if using r^2 is appropr-
iate in this scenario.

>>> `explained_var(model(feats), targs)` #doctest:+SKIP

We can also pick 20 test points not seen during training.

>>> `xss = 40*torch.rand(20)-80/3`
>>> `xss_test = xss.unsqueeze(1)`
>>> `yss = torch.normal(2*xs*torch.cos(xs/10)-5, 10.0)`
>>> `yss_test = yss.unsqueeze(1)`
>>> `explained_var(model(xss_test), yss_test)` #doctest:+SKIP

These r^2 values jump around, over multiple runs of the code
above. Sometimes r^2 is around 0.5 which means that the reqres-
sion poly captures about 50% of the variation in the data. But
sometimes it turns up negative.

Try running the commandline program `dulib_polyreg` which dis-
plays the (sample) point cloud along with the regression poly.
The r^2 values are also computed and displayed.

How can r^2 be negative? What is r^2 actually computing in the
case of polynomial regression? Why is r^2 never negative in the
case of a degree 1 poly (i.e., a line) if we assume a model
that has converged to actual min of the loss function?

For the (ols) regression line, r^2 computes the proportion of
the variation of the data explained by the regression line `over`
`and above that explained by a horizontal line` (with intercept
the mean of the `y` values of the data).

A central question arises: should we even use poly regression
and, if so, what's the best degree?

Note: Another commandline program, `dulib_polyreg_anim`, is pro-
vided that plays an animation of the regression polynomial con-
verging.  (The data is generated by adding i.i.d. noise to the
outputs of f(x) = 2x*cos(x/10)-5, as in the last demonstration.

                    _____________________


!Nonlinear regression!

Let us generate the data as in last demonstration.

>>> `xss = 40*torch.rand(20)-80/3`
>>> `yss = torch.normal(2*xss*torch.cos(xss/10)-5, 10.0)`
>>> `feats = xss.unsqueeze(1); targs = yss.unsqueeze(1)`

The following class has a single hidden layer.

>>> `class NonLinModel(nn.Module):`
...   `def __init__(self, n_hidden):`
...     `super().__init__()`
...     `self.layer1 = nn.Linear(1, n_hidden)`
...     `self.layer2 = nn.Linear(n_hidden, 1)`
...   `def forward(self, xss):`
...     `xss = torch.relu(self.layer1(xss))`
...     `return(self.layer2(xss))`

Let us use 8 nodes in our hidden layer:

>>> `model = NonLinModel(n_hidden = 8)`,

Or, equivalently, using `DUlib`:

>>> model = DenseFFNet(1, 1, (8,))

>>> `model=train(`
...     `model = model`,
...     `crit = nn.MSELoss()`,
...     `train_data = (feats, targs)`,
...     `learn_params = {'lr': 1e-5, 'mo': 0.98}`,
...     `epochs = 8000`,
...     `graph = 0,`
...     `verb = 0)`

>>> `explained_var(model, (feats, targs))` #doctest:+SKIP

>>> `xss = 40*torch.rand(20)-80/3`
>>> `xss_test = xss.unsqueeze(1)`
>>> `yss = torch.normal(2*xss*torch.cos(xss/10)-5, 10.0)`
>>> `yss_test = yss.unsqueeze(1)`
>>> `explained_var(model, (xss_test, yss_test))` #doctest:+SKIP

'''
import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tkinter
import du.lib as dulib
import du.utils
import du.models

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

# if someone is running in WSL, try to catch if no Xserver running
_not_WSL =  not 'WSL_DISTRO_NAME' in os.environ
_has_display = 'DISPLAY' in os.environ
_display = True if _not_WSL or _has_display else False
# This doesn't work currently in WSL. Disabling:
_display = True


def simple_linear_regression():
  """Commandline program that graphs a regression line."""
  xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 20.0)
  xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  d = dulib.optimize_ols(xss)
  parser = argparse.ArgumentParser(
      description = 'Simple linear regression via gradient descent',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-lr', type=float, help='learning rate',
      default=du.utils.format_num(d['lr']))
  parser.add_argument('-mo', type=float, help='momentum',
      default=du.utils.format_num(d['mo']))
  parser.add_argument('-epochs', type=int, help='epochs', default=200)
  h_str = '1 to show graph of loss during training'
  parser.add_argument('-gr', type=int, help=h_str, default=0)
  args = parser.parse_args()

  model = dulib.train(
      model = du.models.SimpleLinReg(),
      crit = nn.MSELoss(),
      train_data = (xss, yss),
      learn_params = {'lr':args.lr, 'mo':args.mo},
      epochs = args.epochs,
      graph = args.gr,
      verb = 2,
      gpu = (-2,))
  params = list(model.parameters())
  slope = params[0].item(); intercept = params[1].item()

  if _display:
    fig, _ = plt.subplots()
    plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
    plt.scatter(xs.tolist(),ys.tolist(),s=9)
    xs = torch.arange(101.)
    plt.plot(xs, 2*xs+9, c='black', lw=.5, label='y = 2x + 9')
    plt.plot(xs, slope*xs+intercept, c='red', lw=.9,\
        label='reg. line: y = {:.2f}x + {:.2f}'.format(slope, intercept))
    plt.legend(loc=0);
    plt.show()
  else:
    print('no X server running')

  print('r^2 for the original data: {}'.\
      format(dulib.explained_var(model(xss),yss)))
  xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 20.0)
  xss_test = xs.unsqueeze(1); yss_test = ys.unsqueeze(1)
  print('r^2 on 20 new test points: {}'.\
      format(dulib.explained_var(model(xss_test),yss_test)))

def simple_linear_regression_animate():
  """Program that plays a regression line animation."""
  assert _display, 'no X-server found'

  xs = 100*torch.rand(40); ys = torch.normal(2*xs+9, 50.0)
  xss = xs.unsqueeze(1); yss = ys.unsqueeze(1)

  d = dulib.optimize_ols(xss)
  parser = argparse.ArgumentParser(
      description = 'Simple linear regression via gradient descent',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-lr',type=float,help='learning rate',
      default=du.utils.format_num(d['lr']))
  parser.add_argument('-mo',type=float,help='momentum',
      default=du.utils.format_num(d['mo']))
  parser.add_argument('-epochs',type=int,help='epochs',default=200)
  parser.add_argument('-bs',type=int,help='batchsize <=40',default=-1)
  args = parser.parse_args()

  plt.ion(); fig, _ = plt.subplots()
  plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
  xs_ = torch.arange(101.)
  plt.plot(xs_, 2*xs_+9, c='black', lw=.5, label='y = 2x + 9')
  plt.scatter(xs.tolist(),ys.tolist(),s=9)

  model = du.models.SimpleLinReg()
  for epoch in range(args.epochs):
    model = dulib.train(
        model = model,
        crit = nn.MSELoss(),
        train_data = (xss, yss),
        learn_params = {'lr':args.lr, 'mo':args.mo},
        epochs = 1,
        bs = args.bs,
        verb = 0,
        gpu = (-2,))
    params = list(model.parameters())
    slope = params[0].item(); intercept = params[1].item()

    plt.clf()
    plt.title('epoch: {}/{}'.format(epoch+1, args.epochs))
    plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
    plt.scatter(xs.tolist(),ys.tolist(),s=9)
    plt.plot(xs_, 2*xs_+9, c='black', lw=.5, label='y = 2x + 9')
    plt.plot(xs_, slope*xs_+intercept, c='red', lw=.9,\
      label='reg. line: y = {:.2f}x + {:.2f}'.format(slope, intercept))
    plt.legend(loc=0)
    try:
      fig.canvas.flush_events()
    except tkinter.TclError:
      plt.ioff()
      exit()
  plt.ioff()
  plt.show()

def simple_polynomial_regression():
  """Program that displays a regression polynomial."""
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
  parser.add_argument('-epochs',type=int,help='epochs',default=8000)
  parser.add_argument('-gr',type=int,help='1 to show loss',default=0)
  parser.add_argument('-show_opt',\
      help='show optimal learning parameters and quit',action='store_true')
  args = parser.parse_args()

  degree = args.deg
  xss = du.models.polyize(xss, degree)
  print('degree is',degree)

  if args.show_opt:
    print(dulib.optimize_ols(xss, verb=2))
    exit()
  if args.lr < 0 or args.mo < 0:
    learn_params = dulib.optimize_ols(xss, verb=1)
  else:
    learn_params = {'lr':args.lr, 'mo':args.mo}

  model = dulib.train(
      model = du.models.SimpleLinReg(degree),
      crit = nn.MSELoss(),
      train_data = (xss, yss),
      learn_params = learn_params,
      epochs = args.epochs,
      graph = args.gr,
      verb = 2,
      gpu = (-2,0))

  if _display:
    fig, _ = plt.subplots()
    plt.xlabel('x',size='larger');plt.ylabel('y',size='larger')
    plt.scatter(xs.tolist(),ys.tolist(),s=9,
        label='y = 2x*cos(x/10)-{}+10*N(0,1)'.format(v_shift))
    xs_ = torch.arange(float(int(x_width)+1)) - x_width/h_scale;
    plt.plot(xs_, xs_*torch.cos(xs_/10)-v_shift, c='black', lw=.5,\
        label='y = 2x*cos(x/10)-{}'.format(v_shift))
    yhatss = model(du.models.polyize(xs_.unsqueeze(1),degree)).squeeze(1)
    plt.plot(xs_, yhatss.detach(), c='red', lw=.9,
        label='reg. poly (deg={})'.format(degree))
    plt.legend(loc=0);
    plt.show()
  else:
    print('no X server running')

  print('r^2 on the original data: {}'.format(dulib.explained_var(model(xss),yss)))
  xs = 40*torch.rand(20)-80/3
  xss_test = xs.unsqueeze(1)
  ys = torch.normal(2*xs*torch.cos(xs/10)-5, 10.0)
  yss_test = ys.unsqueeze(1)
  print('r^2 on 20 new test points: {}'.\
      format(
          dulib.explained_var(model(du.models.polyize(xss_test,degree)),yss_test)))

def poly_string(coeffs):
  """Rudimentary poly string representation

  Args:
    $coeffs$ (`list`): The coefficients with the constant
        term first.

  Returns:
    `str`.

  >>> `print(poly_string([-5,2,3]))`
  3x^2 + 2x + -5
  """
  coeffs = coeffs[::-1]
  degree = len(coeffs) - 1

  string_rep = ''
  for i, coeff in enumerate(coeffs):
    coeff = du.utils.format_num(coeff)
    if i < degree - 1:
      string_rep += '{}x^{} + '.format(coeff, degree - i)
    elif i < degree:
      string_rep += f'{coeff}x + '
    else:
      string_rep += str(coeff)
  return string_rep

def simple_polynomial_regression_animate():
  """Program that plays a regression polynomial animation."""
  assert _display, 'no X-server found'

  parser = argparse.ArgumentParser(
      description =\
         'Simple poly regression via gradient descent. call this'+
         'with lr set to -1 to try optimal learning parameters.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-deg', type=int,help='degree of poly', default=7)
  parser.add_argument('-n', type=int, help='sample size', default=8)
  parser.add_argument('-lr',type=float,help='learning rate',default=.1)
  parser.add_argument('-mo',type=float,help='momentum',default=.9997)
  parser.add_argument('-epochs',type=int,help='epochs',default=30000)
  help_='train for this prop of epochs before graphing'
  parser.add_argument('-pre_train', type=float, help=help_ ,default=0.3)
  parser.add_argument('-graph_pre',type=int,help='1 to graph loss when pretraining',default=0)
  parser.add_argument('-graph_post',type=int,help='1 to show loss',default=0)
  help_='show optimal learning parameters and quit'
  parser.add_argument('-show_opt', help=help_, action='store_true')
  parser.add_argument('-stand', help="standardize x-values", action='store_false')
  parser.add_argument('-step',type=int,help="step this many epochs when post training",default=100)
  parser.add_argument('-seed', type=int,help="seed", default=111)
  help_='roughly evenly distributed x-values'
  parser.add_argument('-un_even', help=help_, action='store_true')
  parser.add_argument('-l2_penalty',type=float,help="ridge regression",default=0)
  parser.add_argument('-nonlin', help="use a non_linear model", action='store_true')
  parser.add_argument('-cloud', help="show only point cloud", action='store_true')
  args = parser.parse_args()

  if args.seed > 0:
      torch.manual_seed(args.seed)

  n = args.n
  epochs = args.epochs
  epochs = 1000 if epochs == 30000 and args.nonlin else args.epochs
  ep_step = args.step
  ep_step = 1 if ep_step == 100 and args.nonlin else args.step
  pre_train = args.pre_train
  pre_train = 0 if pre_train == .3 and args.nonlin else args.pre_train
  cloud = args.cloud
  pre_train = 0 if pre_train == .3 and args.cloud else pre_train
  ep_step = 1 if ep_step == 100 and args.cloud else ep_step
  epochs = 100 if epochs == 30000 and args.cloud else epochs

  x_width = 110.0; h_scale = 1.5; err = 20

  if args.un_even:
    xs = x_width * torch.rand(n) - x_width/h_scale
  else:
    xs = x_width * torch.arange(n)/n + 5*torch.rand(n) - x_width/h_scale

  ys = torch.normal(2*xs/(1.5+torch.cos(xs/10)), err)
  xss = xs.unsqueeze(1)
  yss = ys.unsqueeze(1)

  degree = args.deg

  if args.nonlin:
    model = du.models.DenseFFNet(degree, 1, (8,))
  else:
    model = du.models.SimpleLinReg(degree)
  crit = nn.MSELoss()
  #crit = nn.SmoothL1Loss()
  print(model)
  nm = model.numel_()

  xss = du.models.polyize(xss, degree)
  if args.stand:
    xss, xss_means = dulib.center(xss)
    xss, xss_stdevs = dulib.normalize(xss)
    yss, yss_means = dulib.center(yss)
    yss, yss_stdevs = dulib.normalize(yss)

  if args.show_opt:
    print(dulib.optimize_ols(xss, verb=2))
    exit()
  if args.lr < 0 or args.mo < 0:
    learn_params = dulib.optimize_ols(xss, verb=1)
    lr_ = du.utils.format_num(learn_params['lr'])
    mo_ = round(learn_params['mo'], 10)
    print(f"optimal lr: {lr_}, mo: {mo_}")
  else:
    learn_params = {'lr':args.lr, 'mo':args.mo}
    print(f"learning rate {args.lr}; momentum {args.mo}")

  learn_params = torch.optim.SGD(
      model.parameters(),
      lr = learn_params['lr'],
      momentum = learn_params['mo'],
      weight_decay = args.l2_penalty
  )

  #learn_params = torch.optim.Adam(
  #    model.parameters(),
  #    lr = learn_params['lr'],
  #    weight_decay=.001
  #)

  model = dulib.train(
      model = model,
      crit = crit,
      train_data = (xss, yss),
      learn_params = learn_params,
      epochs = int(pre_train*epochs),
      graph = args.graph_pre,
      verb = 3 if pre_train else 0,
      gpu = (-2,))

  if args.graph_post:
    model = dulib.train(
        model = model,
        crit = crit,
        train_data = (xss, yss),
        learn_params = learn_params,
        epochs = int((1-pre_train)*epochs),
        graph = args.graph_post,
        verb = 3,
        gpu = (-2,))
    exit()

  plt.ion();
  fig, _ = plt.subplots(figsize=(10,7));
  plt.rc('font',size=14)
  plt.xlabel('x',size='large'); plt.ylabel('y',size='large')
  delta = 1
  start=min(xs)-delta; end=max(xs)+delta; step=(end-start)/100
  xs_ = torch.arange(start, end + step, step)
  plt.plot(xs_, 2*xs_/(1.5+torch.cos(xs_/10)), c='black', lw=1,
      label=f'y = 2x/(3/2+cos(x/10))')
  plt.scatter(xs.tolist(),ys.tolist(),s=30,label=f'y = 2x/(3/2+cos(x/10) + {err}*N(0,1)')

  for epoch in range(round((1-pre_train)*epochs)//ep_step+1):
    model = dulib.train(
        model = model,
        crit = crit,
        train_data = (xss, yss),
        learn_params = learn_params,
        epochs = ep_step,
        verb = 0,
        gpu = (-2,))
    if args.stand and not args.nonlin:
      params = list(model.parameters())
      coeffs = [params[-1].item()]
      for param in params[0].squeeze(0):
        coeffs.append(param.item())

    plt.clf()
    total_epoch = round(pre_train*epochs + ep_step*epoch)
    plt.title(
        f'training {nm} weights on {n} features; epoch: {total_epoch}/{epochs}',
        size = 'large'
    )
    plt.xlabel('x',size='large'); plt.ylabel('y',size='large')
    if not cloud:
        plt.plot(xs_, 2*xs_/(1.5+torch.cos(xs_/10)), c='black', lw=1,
            label=f'y = 2x/(3/2+cos(x/10))')
    plt.scatter(xs.tolist(),ys.tolist(),s=30,
        label=f'y = 2x/(3/2+cos(x/10)) + {err}*N(0,1)'
    )
    with torch.no_grad():
      xss_ = du.models.polyize(xs_.unsqueeze(1), degree)
      if args.stand:
        xss_, _ = dulib.center(xss_, xss_means)
        xss_, _ = dulib.normalize(xss_, xss_stdevs)
        yss_, _ = dulib.center(xss_, yss_means)
        yss_, _ = dulib.normalize(xss_, yss_stdevs)
        yhatss = yss_stdevs * model(xss_).squeeze(1) + yss_means
      else:
        yhatss = model(xss_).squeeze(1)

    if not args.stand:
      poly_str = poly_string(coeffs)
      if len(poly_str) <= 45:
          poly_str = poly_str + ' '*(45-len(poly_str))
      else:
          while len(poly_str) > 45:
              poly_str = ' '.join(poly_str.split()[:-2]) + ' ...'
      #plt.plot(xs_, yhatss.detach(), c='red', lw=.9, label=poly_str)
      #leg = plt.legend(loc=8)
      #for t in leg.get_texts():
      #    t.set_ha('left')
    else:
      poly_str = f"piecewise O(x^{degree})" if args.nonlin else f"O(x^{degree})"
      if args.l2_penalty:
          poly_str = poly_str + f"; reg. with {args.l2_penalty}*L2"
    plt.plot(xs_, yhatss, c='red', lw=1.5, label=poly_str)
    if not cloud:
        plt.legend(loc=0)

    #delta = 1
    #plt.xlim(min(xs) - delta, max(xs) + delta)

    try:
      fig.canvas.flush_events()
    except tkinter.TclError:
      plt.ioff()
      exit()

  plt.ioff()
  plt.show()

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
