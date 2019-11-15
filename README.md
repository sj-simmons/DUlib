<p align="right"> <b> DL@DU </b> </p> <a id="dldu"></a>

### DUlib
---

#### Quickstart

At the command line install the latest release:

``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git --user
```

Then have a look at the documentation for the core library:

``` bash
pydoc3 du.lib
```

Suppose you decide to use, in our program, the `train` function from the DUlib:

``` python
from du.lib import train
```

Want to see what is available in DUlib in addition to the core library:
``` bash
pydoc3 du
```

Note: If you are in the Python interpreter, you can
get help on say the `train` function:
``` python
>>> from du.lib import train
>>> help(train)
```

You can find entire programs that use the functionality of DUlib at the DL@DU Project.

---

#### Release information

You can use `pip3` to check the version of DUlib that is installed on your
system by issuing the following at the command line:
``` bash
pip3 show DUlib
```
You can, in  addition, see the files that are installed with:
``` bash
pip3 show -f DUlib
```

**Version 0.1** (March 2019)
  * If you were part of the DL@DU project before about November 1, 2019 and
    you used any of the libraries in the old repo then you can install version 0.1
    ``` bash
    pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.1  --user
    ```
    and have all your code work as you wrote originally wrote it, with the sole
    exception of modifying the way you import functions from library. Now you do,
    for example:
    ``` python
    from du.lib import center, normalize, train, confusion_matrix
    ```
**Version 0.2** (November 6, 2019)
  * A library for NLP applications, [wordlib.py](du/wordlib.py), has been added.

**Version 0.3** (November 14, 2019)
  * Quick install: `pip3 install git+https://github.com/sj-simmons/DUlib.git --user`
  * The API, particularly in `du.lib`, should be stable as of this release, so you
    should go ahead and upgrade. However, in transitioning from earlier versions to 0.3,
    one needs to do some refactoring. For example:
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
      then, in version 0.3, you would use
      ``` python
      from du.lib import train
      ...
      train(model, crit, (xss, yss), ... , mo = args.mo, ...)
      ```
      In other words, some of the keywords for arguments passed to `train` have
      been shortened. Also, you now *must* use key-worded arguments for certain
      arguments (see the documentation).
  * A library called `du.models` has been added that contains various convolutional
    and recurrent models that we use at the DL@DU Project.
  * All of the code in the DL@DU projects has been refactored to version 0.3.

---

#### Experimenting with, building on top of, or otherwise modifying DUlib

If you want to experiment with the code in the libraries, then you can clone
to a local repo on your machine:
``` bash
git clone https://github.com/sj-simmons/DUlib.git@v0.3 --user
```
Suppose you clone to a local repo and modify or add to the code in the libraries, and
then you want to install (directly from your local repo) the modified libraries
to your local machine:
``` bash
cd DUlib
pip3 install -e .
```
Now suppose that you want to revert back to the latest release: 
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git --user
```
