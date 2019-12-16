__doc__ =\
"""
tools from `The DL@DU Project`.

The |module structure|, along with the (non-helper) $functions$ and
$classes$, are, as of this release,

  |du|
  ├── |lib.py|  !the core library!
  │   ├── $center$         mean-center some data
  │   ├── $normalize$      normalize data
  │   ├── $coh_split$      split out testing data
  │   ├── $train$          train a model
  │   ├── $cross_validate$ cross-validate train a model
  │   ├── $confusion_matrix$
  │   ├── $r-squared$
  │   ├── $optimize_ols$   find optimal lr and momentum
  │   ├── $LearnParams_$   class for adding training hyper-params
  │   └── $Momenum$        add momentum to SGD
  ├── |models.py|  !model classes for feed-forward nets!
  │   ├── $denseFFhidden$  compose dense layers
  │   ├── $polyize$        for making polynomials
  │   ├── $SimpleLinReg$   fit polys to 2d point clouds
  │   ├── $FFNet_$         base class for feed-forward nets
  │   └── $DenseFFNet$     factory for dense FF nets
  ├── |examples.py|
  ├── |conv|  !convolutional tools for images and more!
  │   └── |models.py|
  ├── |rec|   !recurrence and natural language processing!
  │   ├── |lib.py|
  │   │   ├── |ngrams.py|  model n-grams for a word corpus
  │   │   └── |cbow.py|    model CBOW wordvecs given a corpus
  │   ├── |examples.py|
  │   └── |models.py|
  └── |util.py|  !potentially helpful utilites!
      └── $stand_args$     easily set command-line options

Example of importing `DUlib`'s functionality into a program:
  ...
  `import du.lib as dulib`
  `from du.models import DenseFFNet`
  ...

Then, in your code, do, say,
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
                    _____________________

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

A good place to begin building your expertise is by reading the
documentation in |examples.py|, which you do by issuing

  |pd du.examples|

at your command line.

In fact, our goal is to provide demonstrations for most all of
the basic over-arching functionality in `DUlib`. Hence, examples
can also be found in the submodules of DUlib. Type, e.g.,

  |pd du.conv.examples|    or    |pd du.rec.examples|

to see examples of involving, respectively, ~convolutional~ and
~recurrent~ nets.

(The `pd` command simply runs a customized version of `pydoc3.5.py`
from the `Python3` standard library.)

                 _____________________


Many of the demonstrations in `DUlib` come with visualizations
(assuming that you have `matplotlib` installed). All of these can
be run from the command line.

For example you can run an animation of gradient descent by is-
suing the command

  |dulib_linreg_anim|

or, to see grad. desc. in action in a more general setting,

  |dulib_polyreg_anim|

Try typing at your command line

  |dulib|<TAB><TAB>

which means type the letters `dulib` and then hit the `TAB` key
twice. You will see all of the available visualizations. Alter-
natively, just start reading the docs.

                    _____________________


A technical note on $breaking out of charset encoding hell$

If you can easily and normally read the trailing phrase in the
last sentence, then you are not even in charset encoding hell.
In this case, you can simply $go ahead and start Deep Learning!$.

However, if that line (and, in fact, some other words/phrases
in the discussion above) is enclosed in boxes or other weird-
ness, then you need to break out of so-called charset encoding
hell. There are a number of escape routes around such rendering
issues; here is one:

There is a good chance that the problems your are experiencing
are due to your being in IDLE (which is the IDE that ships with
Python) or some other IDE that doesn't play nice with `ansi`
escape sequences.

Recommendation: consume the documentation for `DUlib` by using
the `pd` command as described above. That way you can enjoy a
few enhancements like bolding and colorizing of certain words.
This aids in quickly and easily finding requisite information
build with the tools in this library.
"""
import du.util

__author__ = 'Scott Simmons'
__status__ = 'Development'
__date__ = '12/16/19'
__version__ = '0.9'
__copyright__ = """
  Copyright [2019] Scott Simmons

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
__doc__ = du.util._markup(__doc__)
