__doc__ =\
"""
tools from `The DL@DU Project`.

One can use color highlighting when reading the documetation
for this module by using the -c switch when calling pd, e.g.:

  pd -c du

If you see escape characters around words in the tree diagram
below, exit reading this and, at your commandline, issue the
following command. Troubleshoot and return here.

  pd -c du.highlight

The |module| structure, along with the (non-helper) $functions$ and
$classes$ are, as of this release,

  |du|
  ├─ |lib| !the core library!
  │  │  ~data:~
  │  ├─ $coh_split$       split data
  │  ├─ $split_df$        split a dataframe
  │  ├─ $center$          mean-center some data
  │  ├─ $normalize$       normalize data
  │  ├─ $standardize$     mean-center and normalize
  │  ├─ $online_means_stdevs$ online compute means and st devs
  │  ├─ $Data$            easily augment data
  │  ├─ $RandomApply$     apply same transforms on in/output
  │  ├─ $Data2$           adaptively augment data
  │  ├─ $RandomApply2$    differnt transforms on in/output
  │  ├─ $FoldableData$    cross-validation with DataLoaders
  │  │  ~training:~
  │  ├─ $train$           train a model
  │  ├─ $cv_train$        cv (cross-validate) train a model
  │  ├─ $cv_train2$       cv train a model on augmented data
  │  ├─ $LearnParams_$    class to add training parameters
  │  ├─ $Momentum$        add momentum to SGD
  │  ├─ $optimize_ols$    optimize learning rate and momentum
  │  │  ~metrics:~
  │  ├─ $explained_var$   compute the explained variance
  │  └─ $class_accuracy$  compute accuracy of classification
  │                     (optionally, displays confusion mat.)
  ├─ |models| !functions and classes for feed-forward nets!
  │  ├─ $denseFFhidden$   compose dense layers
  │  ├─ $polyize$         for making polynomials
  │  │  ~model classes:~
  │  ├─ $FFNet_$          base class for feed-forward nets
  │  ├─ $SimpleLinReg$    fit polys to 2d point clouds
  │  └─ $DenseFFNet$      factory for dense FF nets
  ├─ |examples|
  ├─ |conv| !convolutional tools for images and more!
  │  ├─ |models|
  │  │  ├─ $metalayer$    make a conv2d plus max_pooling layer
  │  │  ├─ $convFFhidden$ make a composition of metalayers
  │  │  ├─ $ConvFFNet$    factory for convolutional nets
  │  │  ├─ $OneMetaCNN$   one metalayer, one dense conv net
  │  │  └─ $TwoMetaCNN$   two metalayer, two dense conv net
  │  └─ |examples|
  ├─ |rec| !recurrence and natural language processing!
  │  ├─ |lib|
  │  │  ├─ |ngrams|       model n-grams for a word corpus
  │  │  └─ |cbow|         model CBOW wordvecs given a corpus
  │  ├─ |models|
  │  └─ |examples|
  ├─ |utils| !potentially helpful utilites!
  │  ├─ $standard_args$   easily set command-line options
  │  ├─ $args2string$     easily make a logging string
  │  ├─ $print_devices$   print the devices available
  │  └─ $get_device$      get a device
  └─ |highlight|          highlighting conventions for docs

!Quick Start!

Import `DUlib`'s functionality into your program with something
like:

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

Familiarize yourself with DUlib by reading through some core
usage examples. From your command line:

  `pd du.examples`

Our goal is to provide demonstrations for most all basic funct-
ionality in `DUlib`. Hence, examples of usage can be found in the
submodules of DUlib: type, e.g., `pd du.conv.examples` or `pd du.`
`rec.examples`. to see demonstrations involving ~convolutional~ and
~recurrent~ nets.

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
aic -- lies in giving machines nearly absolute freedom in their
efforts to discover patterns in data.

This freedom combined with the enormity of data available in
the modern information era, has fueled the resurgence in viabi-
lity of neural networks.

None of this would be possible without artfully efficient imp-
lementations of the massive ~chain rule~ computations driving the
convergence of deep networks. Those algorithms comprise the me-
thod called ~gradient descent~, the ~stochastic~ version of which
is the workhorse of machine learning.

As late as the 90s, some doubted the computational feasibility
of deep neural networks. Yet, not only do multilayered of net-
works (and even ~recurrent~ networks) converge (when tuned appro-
priately), they, in some cases, reveal windows into dimensions
of big data that were previously unobserved.

Cutting-edge techniques in Deep Learning even put machines at
liberty to conceptualize parts of their own architectures.

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

"""
import du.utils

__author__ = 'Scott Simmons'
__status__ = 'Development'
__date__ = '10/28/24'
__version__ = '0.9.96'
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
#__doc__ = du.utils._markup(__doc__)
