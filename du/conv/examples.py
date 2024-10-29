#!/usr/bin/env python3
"""demonstrate the functionality of `du.conv`.

\n!Classifying MNIST Digits!

Here we work with the well known MNIST dataset, which comprises
60K, 28x28, gray-scale, hand-drawn digits. Our first demonstra-
tion works with a randomly selected 6K examples, a size approp-
riate for any machine.

The second demo, which works with the entire dataset, trains in
about 5 minutes on, say, a Core i5. But what if your images are
color and/or larger and you have more of them? In general, how
do we deal with large datasets?

!Demo 1: 6,000 digits!

The MNIST dataset can be downloaded from Yann LeCunn's website
using the `DataLoader` class from `torch.utils.data`. (This requir-
es that you have installed the `torchvision` package.)

The following retrieves the dataset (if it was not previously
downloaded) and puts it in subdirectory named `data` off of your
home directory.

>>> `import torch`
>>> `from torchvision import datasets`
>>> `from torch.utils.data import DataLoader`
>>> `dl = DataLoader(`
...   `datasets.MNIST('~/data/mnist',train=True,download=True))`

We can now easily get out hands on the actual images, but they
are `torch.ByteTensor`s.

>>> `features = dl.dataset.data`
>>> `isinstance(features, torch.Tensor)`
True
>>> `features.type()`
'torch.ByteTensor'

Let us convert `features` to a `float` tensor.

>>> `features = features.to(dtype=torch.float32)`
>>> `features.type()`
'torch.FloatTensor'

How many examples are there in the MNIST data?

>>> len(features)
60000

We can get the 150th image and print out its middle 20x20 part.

>>> `xs = features[149]`
>>> `xs.shape`
torch.Size([28, 28])
>>> `for x in xs[4:24]:`
...   `print(''.join(map(lambda x: str(int(x)).rjust(3)`,
...        `x.tolist()[4:24])))`
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0101144253253255253253253253255128 25  0  0  0
  0  0 57163226249252252252253252252252252253252103  0  0  0
 19166234252253252252214195196 70133 71227253252195  0  0  0
 76246252252225 99 84 28  0  0  0  0  0140253252195  0  0  0
  0 88112112  0  0  0  0  0  0  0  0 95203253252195  0  0  0
  0  0  0  0  0  0  0  0  0  0  0120253253255215 31  0  0  0
  0  0  0  0  0  0  0  0  0 38210246252252215 33  0  0  0  0
  0  0  0  0  0  0  0  0 32222252252252173 31  0  0  0  0  0
  0  0  0  0  0 26 60169215253252252252252169 34  0  0  0  0
  0  0  0  0 63240252252252253252252252252253214 31  0  0  0
  0  0  0  0255253253253190141 94  0  0141255253 56  0  0  0
  0  0  0  0 90167167 74 12  0  0  0  0 94253252180  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0253252195  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0253252195  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0 79253252195  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0141255253196  0  0  0
  0  0  0 48 85  9  0  0  0  0  0 67178240253223 52  0  0  0
  0  0  0140253203165 57135198197240252252228 52  0  0  0  0
  0  0  0 47237252252252252253252252249145 47  0  0  0  0  0

Next let us get the targets, which are also `ByteTensor`s. and
so convert those to type `LongTensor`.

>>> `targets = dl.dataset.targets.to(dtype=torch.long)`
>>> `targets.type()`
'torch.LongTensor'

Unsurprisingly, we have

>>> `targets[149]`
tensor(3)

!Building a classifier!

In order to run a CPU, we work with a randomly selected 1/10 of
the data.

>>> `import du.lib as dulib`
>>> `feats, _, targs, _ = dulib.coh_split(0.1,features,targets)`

We can use `DUlib` to quickly build a model.

>>> `from du.conv.models import ConvFFNet`
>>> `model = ConvFFNet(`
...     `in_size = (28, 28)`,
...     `n_out = 10`,
...     `channels = (1, 16)`,
...     `widths = (10,))`
>>> `model`
ConvFFNet(
  (conv): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),...)
      (1): BatchNorm2d(16,...)
      (2): ReLU()
      (3): MaxPool2d(kernel_size=2, stride=2, padding=0, ...)
    )
  )
  (dense): Sequential(
    (0): Linear(in_features=3136, out_features=10, bias=True)
    (act0): ReLU()
    (lin1): Linear(in_features=10, out_features=10, bias=True)
  )
)

We split out 1/6 of the data for testing.

>>> `splits = dulib.coh_split(5/6, feats, targs)`
>>> `train_feats, test_feats, train_targs, test_targs = splits`

Now center and normalize the features.

>>> `train_feats, train_means = dulib.center(train_feats)`
>>> `train_feats, train_stdevs = dulib.normalize(train_feats)`

And center and normalize the test data w/r to the training data
means and standard deviations.

>>> `test_feats, _ = dulib.center(test_feats, train_means)`
>>> `test_feats, _ = dulib.normalize(test_feats, train_stdevs)`

Train the model

>>> `import torch.nn as nn`
>>> `model = dulib.train(`
...     `model = model`,
...     `crit =  nn.NLLLoss()`,
...     `train_data = (train_feats, train_targs)`,
...     `test_data = (test_feats, test_targs)`,
...     `learn_params = {'lr' : 0.001, 'mo': 0.92}`,
...     `bs = 20`,
...     `epochs = 15`,
...     `verb = 0)`# doctest:+SKIP

Let us check accuracy on the test data.

>>> `0.94 < dulib.confusion_matrix(`
...     `model(test_feats)`,
...     `test_targs`,
...     `torch.arange(10))`# doctest:+SKIP
True
                    _____________________

For convenience, below is the code in this demo in the form of
a program:

`import torch`
`import torch.nn as nn`
`import du.lib as dulib`
`from du.conv.models import ConvFFNet`
`from torchvision import datasets`
`from torch.utils.data import DataLoader`

`dl = DataLoader(`
    `datasets.MNIST('mnist',train=True,download=True))`

`features = dl.dataset.data.to(dtype=torch.float32)`
`targets = dl.dataset.targets.to(dtype=torch.long)`

`feats, _, targs, _ = dulib.coh_split(0.1,features,targets)`

`train_feats, test_feats, train_targs, test_targs = \`
    `dulib.coh_split(5/6, feats, targs)`

`train_feats, train_means = dulib.center(train_feats)`
`train_feats, train_stdevs = dulib.normalize(train_feats)`

`test_feats, _ = dulib.center(test_feats, train_means)`
`test_feats, _ = dulib.normalize(test_feats, train_stdevs)`

`model = dulib.train(`
    `model = ConvFFNet(`
        `in_size = (28, 28),`
        `n_out = 10,`
        `channels = (1, 16),`
        `widths = (10,)),`
    `crit =  nn.NLLLoss(),`
    `train_data = (train_feats, train_targs),`
    `test_data = (test_feats, test_targs),`
    `learn_params = {'lr' : 0.001, 'mo': 0.92},`
    `bs = 20,`
    `epochs = 15,`
    `gpu = (-1,))`

`print('accuracy on test data:', dulib.confusion_matrix(`
    `(model, test_feats), test_targs, torch.arange(10)))`


!Demo 2: 60,000 digits!

We will now work with the entire MNIST dataset. For large data-
sets, a practical problem arises: what if we have more examples
in our data than will fit in our GPU? (Here, we are assuming
access to a machine with a GPU, though the functionality we im-
plement will of course run on a CPU.)

Executing the code above (in Demo 1) on all 60,000 MNIST digits
causes an 8GB GPU to run out memory. On the other had, we typi-
cally use a small batch-size when training. The size of the mo-
del (i.e, the number of weights) is determined by the architec-
ture.

The one-metalayer model in Demo 1 has 31896 weights (`float`s) in
total. If we use 32 bit floats, then this is about 128K bytes.
Of course, more RAM (or VRAM, on a GPU) is required in order to
carryout the calculus of back-propagation.  Still, at issue is
not the size of the model; it is the size of the data.

How can we carve out chunks of data, move those to the GPU, and
foward/backpropagate them through the model during training. We
could, for instance, chunk out `N` mini-batches at a time, Let `bs`
denote the mini-batch size.

The features of such a chunk of MNIST data constitute `N`*`bs`*28*
28*4 = 3136*`N`*`bs` bytes of data, and the targets, `N`*`bs`*8 bytes;
so 3144*`N`*`bs` bytes. If `N`*`bs` = 60K, then this just computes the
size of the MNIST data, which, in MB, is 3144*60000/1024**2,
about 180 MB.

Let us assume that the MNIST data has been downloaded and that
an instance of `DataLoader` named `dl` holds all of the data. See
the beginning of the Demo 1 for details.

>>> `features = dl.dataset.data.to(dtype=torch.float32)`
>>> `targets = dl.dataset.targets.to(dtype=torch.long)`

Let us now treat `features` and `targets` as we would any fairly
large dataset, and see how we can use `torch.utils.data` to our
advantage (forgetting, in other words, that we got the data in
`features` and `targets` in the first place by employing `torch.uti`
`ls.data.DataLoader`).

The data `features` and `targets` are just tensors.

>>> `features.shape; targets.shape`
torch.Size([60000, 28, 28])
torch.Size([60000])

Below, we construct a new `dataloader` instance, having the prop-
erties described above. First, we need an appropriate instance
of `torch.utils.data.Dataset`.

Minimally, one implements  `__len__` and `__getitem__` methods when
extending `Dataset`.

>>> from torch.utils import data
>>> class TrainingDataset(data.Dataset):
...   '''A class for training data.'''
...   def __init__(self, feats, targs):
...     '''Constructor'''
...     assert len(feats) == len(targs)
...     self.feats = feats
...     self.targs = targs
...   def __len__(self):
...     '''Return total number of training examples'''
...     return len(feats)
...   def __getitem__(self, item):
...     '''Return a single training example.'''
...     return self.feats[item], self.targs[item]

Let us now split out 5/6 of 60K examples for training.

>>> `splits = dulib.coh_split(5/6, features, targets)`
>>> `train_feats, test_feats, train_targs, test_targs = splits`

As before, we center and normalize the features.

>>> `train_feats, train_means = dulib.center(train_feats)`
>>> `train_feats, train_stdevs = dulib.normalize(train_feats)`

And center and normalize the test data w/r to the training data
means and standard deviations.

>>> `test_feats, _ = dulib.center(test_feats, train_means)`
>>> `test_feats, _ = dulib.normalize(test_feats, train_stdevs)`

Next, we instance the class we defined above.

>>> `train_data = TrainingDataset(train_feats, train_targs)`

We can now take slices; e.g.,

>>> minibatch = train_data[:100]
>>> minibatch[0].shape, minibatch[1].shape,
(torch.Size([100, 28, 28]), torch.Size([100]))

Using PyTorch's DataLoader class, we define

>>> #dataLoader

Train the model

>>> `model = dulib.train(`
...     `model = model`,
...     `crit =  nn.NLLLoss()`,
...     `train_data = (train_feats, train_targs)`,
...     `test_data = (test_feats, test_targs)`,
...     `learn_params = {'lr' : 0.001, 'mo': 0.92}`,
...     `bs = 20`,
...     `epochs = 15`,
...     `verb = 0)` # doctest:+SKIP

Let us check accuracy on the test data.

>>> `0.94 < dulib.confusion_matrix(`
...     `model(test_feats)`,
...     `test_targs`,
...     `torch.arange(10))` # doctest:+SKIP
True

"""
import du.utils

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
