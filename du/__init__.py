__doc__ =\
"""
tools from `The DL@DU Project`.

The |module| structure, along with the (non-helper) $functions$ and
$classes$ are, as of this release,

  |du|
  ├─ |lib.py|     !the core library!
  │  ├─ $center$          mean-center some data
  │  ├─ $normalize$       normalize data
  │  ├─ $coh_split$       split out testing data
  │  ├─ $train$           train a model
  │  ├─ $cross_validate$  cross-validate train a model
  │  ├─ $confusion_matrix$
  │  ├─ $r-squared$
  │  ├─ $optimize_ols$    optimize learning rate and momentum
  │  ├─ $LearnParams_$    class to add training hyper-params
  │  └─ $Momentum$        add momentum to SGD
  ├─ |models.py|  !model classes for feed-forward nets!
  │  ├─ $denseFFhidden$   compose dense layers
  │  ├─ $polyize$         for making polynomials
  │  ├─ $FFNet_$          base class for feed-forward nets
  │  ├─ $SimpleLinReg$    fit polys to 2d point clouds
  │  └─ $DenseFFNet$      factory for dense FF nets
  ├─ |examples.py|
  ├─ |conv|       !convolutional tools for images and more!
  │  ├─ |models.py|
  │  │  ├─ $metalayer$    make a conv2d plus max_pooling layer
  │  │  ├─ $convFFhidden$ make a composition of metalayers
  │  │  ├─ $ConvFFNet$    factory for convolutional nets
  │  │  ├─ $OneMetaCNN$   one metalayer, one dense conv net
  │  │  └─ $TwoMetaCNN$   two metalayer, two dense conv net
  │  └─ |examples.py|
  ├─ |rec|        !recurrence and natural language processing!
  │  ├─ |lib.py|
  │  │  ├─ |ngrams.py|    model n-grams for a word corpus
  │  │  └─ |cbow.py|      model CBOW wordvecs given a corpus
  │  ├─ |models.py|
  │  └─ |examples.py|
  └─ |utils.py|    !potentially helpful utilites!
     └─ $stand_args$     easily set command-line options

Note: consume the examples by typing, e.g., `pd du.examples` or
`pd du.conv.examples`.

To import `DUlib`'s functionality into your program, consider do-
ing something like:

  ...
  `import du.lib as dulib`
  `from du.models import DenseFFNet`
  ...
  `xss, _ = dulib.center(xss)`
  ...
  `model = DenseFFNet(`
      ...
  `)`
  `model = dulib.train(`
      `model`,
      `crit`,
      `train_data = (xss, yss)`,
      ...
  `)`
  ...

!Overview!

Our goal in writing this library is two-fold. We wish to pro-
vide well-designed, high-quality tools that package the power
and extensibility of `PyTorch`, with which we can create our ex-
periments and designs in code in near real-time.

We also hope to provide a concise vehicle by which we may hone
our command of Machine/Deep Learning and explore the nature of
the way, in fact, not machines, but humans, wield such tools as
they unearth the secrets held by data.

The true power of Deep Learning -- which, since it is formulat-
ed in terms of neural nets, was once thought to be rather pros-
aic -- lies in giving machines nearly absolute freedom to dis-
cover patterns in data.

This freedom combined with the enormity of data available in
the modern information era, has fueled the resurgence in viabi-
lity of neural networks.

None of this would be possible without artfully efficient imp-
lementations of the massive ~chain rule~ computations driving the
convergence of deep networks. Those algorithms comprise the me-
thod called ~gradient descent~, the ~stochastic~ version of which
is the workhorse of machine learning.

As late as the 90s, some doubted the computational infeasibil-
ity of deep networks. Yet, not only do multilayered of networks
(and even ~recurrent~ networks) converge (when tuned appropri-
ately), they in some cases produce windows into dimensions of
big data previously unobserved.

Cutting-edge techniques in Deep Learning even put machines at
liberty to conceptualize parts of their own architectures.

!Quick Start!

A good place to begin is by reading/working through some core
examples, which you do by issuing the command `pd du.examples` at
your command line.

Our goal is to provide demonstrations for most all basic funct-
ionality in `DUlib`. Hence, examples of usage can be found in the
submodules of DUlib: type, e.g., `pd du.conv.examples` or `pd du.`
`rec.examples`. to see demonstrations involving ~convolutional~ and
~recurrent~ nets.

                    _____________________


Many of the demonstrations in `DUlib` come with visualizations
(assuming that you have `matplotlib` installed). All of these can
be run from the command line.

For example you can run an animation of gradient descent by is-
suing the command `dulib_linreg_anim` or, to see gradient descent
in action in a more general setting, `dulib_polyreg_anim`.

Try typing at your command line `dulib<TAB><TAB>` which means to
type the letters `dulib` and then hit the `TAB` key twice. You will
see all of the available visualizations. Alternatively, simply
start reading the docs.

                    _____________________


A technical note on !breaking out of charset encoding hell!.

If you can easily and normally read the trailing phrase in the
last sentence, then you are not even in charset encoding hell.
In this case, you can simply go ahead and start deep learning.

However, if that line (and, in fact, other words or phrases in
the discussion above) is enclosed in boxes or other weird char-
acters, then you are in so-called charset encoding hell. There
are a number of ways around such rendering issues.

There is a good chance that the problems your are experiencing
are due to your being in IDLE (which is the IDE that ships with
Python) or some other IDE that doesn't play nice with `ansi` es-
cape sequences.

Recommendation: consume the documentation for `DUlib` by using
the `pd` command at the command-line as described above. That way
you can enjoy a few enhancements like bolding and colorizing of
certain words. This aids in quickly and easily finding apis for
this library.

Now, if you in fact already are using the command-line and yet
still experiencing char escape hell, then try manually setting
PAGER and TERM environment variables.

A great value for PAGER is the value: less -r. You can set the
PAGER environment variable on *nix with the bash command:

  export PAGER='less -r'

which you can add to the end of ~/.bashrc if you wish. Viable
values for TERM are, for example, any one of: screen-256color,
xterm-256color, or ansi.

You can set TERM in bash with, e.g.: export TERM=xterm-256color

"""
import du.utils

__author__ = 'Scott Simmons'
__status__ = 'Development'
__date__ = '12/29/19'
__version__ = '0.9'
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
__doc__ = du.utils._markup(__doc__)
