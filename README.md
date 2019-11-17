<p align="right"> <b> DL@DU </b> </p> <a id="dldu"></a>

### DUlib
---

#### Quickstart
* At the command line, install the latest release: `pip install DUlib --user`
* Then have a look at functions provided by the core library: `pydoc3 du.lib`.
* And perhaps some examples of their usage: `pydoc3 du.examples`
* Suppose you decide to use the `train` function.
  Then, in your program, put `from du.lib import train`
* Want to see what is available in DUlib in addition to the core
  library: `pydoc3 du`
  * Then maybe `pydoc3 du.conv.models` depending on what you are working on.
* You can find entire programs that use the functionality of `DUlib` at
  the [DL@DU Project](https://github.com/sj-simmons/deep-learning#dldu).
* If you are in the Python interpreter and have imported say the `train`
  function with `from du.lib import train`, then you can get help on `train`
  function with:
  ``` python
  >>> help(train)
  ```
* At the command line, you would issue the command `pydoc3 du.lib.train`
  for help.

---

#### Release information

You can check the version of DUlib that is installed on your system
with: `pip show DUlib`.

**Version 0.1** (March 2019)
  * If you were part of the DL@DU project before about November 1, 2019 and
    you used any of the libraries in the old repo then you can install version 0.1
    ``` bash
    pip install git+https://github.com/sj-simmons/DUlib.git@v0.1  --user
    ```
    and have all your code work as you originally wrote it, with the sole
    exception of modifying the way you import functions from library. Now,
    you would import some basic functions with:
    ``` python
    from du.lib import center, normalize, train, confusion_matrix
    ```
**Version 0.2** (November 6, 2019)
  * A library for NLP applications, [rec/lib.py](du/rec/lib.py), has been added.

**Version 0.3** (November 14, 2019)
  * Quick install: `pip install git+https://github.com/sj-simmons/DUlib.git@v0.3 --user`
  * The API, particularly in `du.lib`, should be stable as of this release, so you
    should go ahead and upgrade. However, in transitioning to version 0.3,
    you need some refactoring. For example:
    * If you were importing `device` in version 0.1 and doing something like
      ``` python
      print('running on', device)
      ```
      you would, in version 0.3 (and onward), do this:
      ``` python
      from du.lib import get_device
      ...
      device = get_device()
      print('running on', device)
      ```
    * If you were using key-worded arguments when calling `train` in version 0.1
      with a line similar to this:
      ``` python
      train(model, criterion, features = xss, targets = yss, learning_rate = 0.1, epochs = 10, batchsize = 20)
      ```
      then, in version 0.3 onward, you would use
      ``` python
      from du.lib import train
      ...
      train(model, crit, train_data = (xss, yss), lr = 0.1 , eps = 10, bs = 20)
      ```
      In other words, some of the keywords for arguments passed to `train` have
      been shortened. Also, you now *must* use key-worded arguments for certain
      arguments. See the documentation with `pydoc3 du.lib.train`.

**Version 0.5** (November 17, 2019)
  * DUlib has been uploaded to [pypi.org](https://pypi.org/) and can now be
    installed be simply issuing the command `pip install DUlib` (or, depending on
    your setup, `pip install DUlib --user`) at the command line.
    * The pypi page is here:
      [pypi.org/project/DUlib/](https://pypi.org/project/DUlib/)
    * To upgrade DUlib type: `pip install -U DUlib`.
  * Function APIs are largely unchanged. An exception is the `graph` argument
    of the train function which is an `int` rather than a boolean.  To enable
    graphing while training (which requires `matplotlib` and a running X server)
    call train with the argument `graph = 1`, or put any positive number less
    than the number of epoch over which you are training.

    Here is an example of graph captured at the end of training. This is picture
    of a dense network training on the digit classification problem.  From the
    graph, it is clear that the network is in danger of over-fitting.  (It is
    best to use a convolutional network for digit classification).

    <p align="center">
      <img height="400" src="graph1.svg">
    </p>

    Putting `graph = 5`, for example, redraws the graph at epoch 5, throwing
    a way the graph at previous epochs, which can be useful in some instances.
  * The modules in the library are now structured as:
    ```bash
        du
        ├── examples.py
        ├── lib.py
        ├── conv
        │   └── models.py
        └── rec
            ├── lib.py
            └── models.py
    ```
  * All code in the DL@DU projects has been refactored to version 0.5.

---

#### Modifying DUlib

If you want to experiment with the code in the libraries, then you can clone
to a local repo on your machine:
``` bash
git clone https://github.com/sj-simmons/DUlib.git --user
```
Suppose you clone to a local repo and modify or add to the code in the libraries,
and then you want to install (directly from your local repo) the modified
libraries to your local machine:
``` bash
cd DUlib
pip install -e .
```
Now suppose that you want to revert back to the latest release:
``` bash
pip install git+https://github.com/sj-simmons/DUlib.git --user
```
or just
``` bash
pip install DUlib --user
```

Please let Simmons know of any new functionality that you implement and wish
to add to DUlib!
