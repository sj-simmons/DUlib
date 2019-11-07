<p align="right"> <b> DL@DU </b> </p> <a id="dldu"></a>

### DUlib
---

#### Versions

**0.1** (March 2019)
  * If you were part of the DL@DU project before about November 1, 2019 and
    you used any of the libraries in the old repo then you can install version 0.1
    and have all your code work as written with the sole exception of modifying
    the way you import functions from library (see below).
  * Quick install: `pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.1`

**0.2** (November 6, 2019)
  * Added [wordlib.py](du/wordlib.py) for NLP applications.
  * The functionality and functions (like `train`) in the original library
    have been enhanced and tweaked.  Slight refactoring of your code might be
    required when transition from version 0.1 to 0.2. Namely:
    * If you were importing `device` in version 0.1 and doing something like
      ``` python
      print('running on', device)
      ```
      you would, in version 0.2, do this:
      ``` python
      from du.lib import get_device
      ...
      device = get_device()
      print('running on', device)
      ```
    * If you were using key-worded arguments when calling `train` in version 0.1
      with a line similar to this:
      ``` python
      train(model, criterion,  features = xss, targets = yss, ... , momemtum = args.mo, ...)
      ```
      then, in version 0.2, you would use
      ``` python
      from du.lib import train
      ...
      train(model, criterion, feats = xss, targs = yss, ... , mo = args.mo, ...)
      ```
      In other words, some of the keywords for arguments passed to `train` have
      been shortened. Also, you now *must* use key-worded arguments for certain
      arguments (see the documentation).
    * All of the code in the DL@DU projects has been refactored to version 0.2.
  * Quick install: `pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.2`

---

#### Documentation

Perusing this repo will only show you the code for the most recent release
version (unless you try to dig through previous commits).  So if you are interested
in the functionality of say version 0.1, Then don't click on the code on this web
page; instead you can do this.
* Install version 0.1 on your system:
  ``` bash
  pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.1
  ```
* Then type `pydoc3 du` at the command line.  You'll see something like this:
  ``` bash
  Help on package du:

  NAME
      du

  PACKAGE CONTENTS
      lib

  FILE
      /home/ssimmons1331/Insync/Code/DUlib/du/__init__.py
  ```
  which tells you that the `du` package that you have installed contains
  a single module called `lib`.

  Then you can type, for example,
  ``` bash
  pydoc3 du.lib
  ```
  and see all the documentation for `du.lib` (which is where basic functions like
  `train` live).  Similarly you can type `pydoc3 du.wordlib` if you have version 0.2
  installed.

  To see the documentation for, say, just the `train` function (below refers to the 0.2
  version of `train`):
  ```
  > pydoc3 du.lib.train
  Help on function train in du.lib:

  du.lib.train = train(model, crit, feats, targs, **kwargs)
      Return the model trained with the given hyper-parameters.

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
  ```
* If you are in the Python interpreter, you can get help on the `train` function:
  ``` python
  >>> from du.lib import train
  >>> help(train)
  ```
---

#### Installation and usage

First, install the library on your system using `pip3`. To install, for example,
version 0.2, issue this command at the command prompt:
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.2
```
(You might need to add the `--user` option to the above command. Never run
pip3 with `sudo`.)

Then, in your program, write, for example:
``` python
from du.lib import center, normalize, train, confusion_matrix
```
or, say, if you have version 0.2 installed,
``` python
from du.wordlib import SimpleRNN
```

You can use `pip3` to check the version of DUlib that is installed on your
system by issuing the following at the command line:
``` bash
pip3 show DUlib
```

You can additionally see the files that are installed with:
``` bash
pip3 show -f DUlib
```
If you want to experiment with the code in the libraries, then you can clone
to a local repo on your machine:
``` bash
git clone https://github.com/sj-simmons/DUlib.git@v0.2
```
Suppose you clone to a local repo and modify or add to the code in the libraries, and
then you want to install (directly from your local repo) the modified libraries
to your local machine:
``` bash
cd DUlib
pip3 install -e .
```
Now suppose that you want to revert back to Simmons' version 0.2:
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.2
```
