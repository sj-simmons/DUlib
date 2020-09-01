<a id="dldu"></a>
<p align="right"> <b> <a href="https://github.com/sj-simmons/deep-learning#dldu"> The DL@DU Project</a> </b> </p>

### DUlib
---

#### Quickstart

* At the command line, install the latest stable release (from [pypi.org](https://pypi.org/project/DUlib/)):
  ``` bash
  pip3 install DUlib --user
  ```
  (In the presence of a previously installed version, try `pip3 install -U DUlib --user`.)

  Or, install the latest development release with, e.g.:
  ``` bash
  pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.9  --user
  ```
* Absolute best practice is to replace the command `pip3` in the commands above (and below)
  with the command `/usr/bin/pip3`; see e.g., [here](https://sj-simmons.github.io/dl/setup/install_packages).

You can now:
* have a look at the module structure of DUlib: `pd du`
* work through some demonstrations that illustrate basic usage: `pd du.examples`

#### Installation notes

By now you are likely aware of [python environment
hell](https://imgs.xkcd.com/comics/python_environment.png).  We recommend simply
*always* installing `DUlib` using the `--user` option with `pip3` &mdash; unless
you are installing system-wide for multiple users, in which case prefacing with
`sudo` might be best.

* At the command line, install the latest release:
  ``` bash
  pip3 install DUlib --user
  ```
  (or `pip3 install -U DUlib --user` in the presence of a previously installed version).

  Notes:
  * If you don't have a `pip3` command then your `pip` command  must in fact
    *be* pip3 (you can check this with `pip -V`).  In this case just use
    ``` bash
    pip install DUlib --user
    ```

  * If you've previously installed  many different versions of DUlib, and especially
    if you've installed from a local cloned version of this repo, then things
    on your system might be kind of a mess. Consider first uninstalling DUlib:
    ``` bash
    pip3 uninstall DUlib
    ```
    and/or forcing an upgrade
    ``` bash
    pip3 install -U  DUlib --force
    ```

  * The most fail-safe way to use pip is
     ``` bash
     python -m pip install ...
     ```
     This is good because you almost certainly have more than one version Python
     installed, and `python` is pointing to whichever version your system thinks
     is *the* python.
  * Mere mortals should never type `sudo pip...`.

Assuming that you installed using `--user`, some DUlib related executables should
now be installed in a local subdirectory below your home directory. On Linux, the
directory is `~/.local/bin/`.

* Type `which pd` at the command line. If you see something
  like `/home/your_username/.local/bin/pd`, then you are in good shape so you
  can skip the next bullet-point.
* If you see nothing, then likely you need to adjust your `PATH` by running the
  following command at the command line:
  ``` bash
  echo "export PATH=$PATH:~/.local/bin" >> ~/.profile
  ```
  Note: your PATH will **not** automatically be updated for the remainder of
  your shell session.  To update without exiting and restarting your shell do:
  ``` bash
  source ~/.profile
  ```

To verify without doubt that DUlib is installed:
* start up the Python3 interpreter and type, say, `import du.lib`. If you see
  no errors, then you are good.


<!---
* Suppose you decide to use the `train` function.
  Then, in your program, put `from du.lib import train`
* To see what is available in DUlib in addition to the core
  library: `pydoc3 du`
  * and then maybe `pydoc3 du.conv.models` depending on what you are working on.
* You can find programs that together demonstrate the full functionality of
  `DUlib` at the [DL@DU Project](https://github.com/sj-simmons/deep-learning#dldu).
* If you are in the Python interpreter and have imported say the `train`
  function with `from du.lib import train`, then you can get help on `train`
  function with: `help(train)`
* At the command line, you would issue the command `pydoc3 du.lib.train`
  for help on `train`.
* The downstream repo for DUlib
    is: [pypi.org/project/DUlib/](https://pypi.org/project/DUlib/).
-->

---

#### Release information

You can check the version of DUlib that is installed on your system
with: `pip show DUlib` (or `pip3 show DUlib`, if necessary)

**Version 0.1** (March 2019).
  * If you were part of the DL@DU project before about November 1, 2019 and
    you used any of the libraries in the old repo then you can install version 0.1
    ``` bash
    pip install git+https://github.com/sj-simmons/DUlib.git@v0.1  --user
    ```
    and have all your code work as you originally wrote it, with the sole
    exception of modifying the way you import functions from the library. Now,
    you would import some basic functions with:
    ``` python
    from du.lib import center, normalize, train, confusion_matrix
    ```
**Version 0.6** (November 17, 2019).
  * DUlib now has a downstream repo on [pypi.org](https://pypi.org/) so that it can now
    be installed by simply issuing the command `pip3 install DUlib` (or, depending
    on your setup, `pip3 install DUlib --user`) at the command line.
    * The pypi page is here:
      [pypi.org/project/DUlib](https://pypi.org/project/DUlib)
    * Your DUlib installation can be upgraded with a command like `pip3 install -U DUlib`
  * The `graph` argument of the train function is now an `int` rather than
    a boolean.  To enable graphing while training (which requires `matplotlib` and
    a running X server) call `train` with the argument `graph = 1`, or put any
    positive number less than the number of epochs over which you are training.

    Here is an example of a graph that was captured at the end of training. This is a picture
    from a dense network training on the digit classification problem.  From the
    graph, it is clear that the network is in danger of over-fitting &mdash; providing
    evidence in support of the mantra that a convolutional network is the preferred flavor
    of network for image classification).

    <p align="center">
      <img height="400" src="images/graph1.svg">
    </p>

    Putting `graph = 5`, for example, when calling `train`, redraws the graph at epoch 5, throwing
    a way the graph at previous epochs, which can be useful in some instances.

**Version 0.8** (November 21, 2019)
  * The new normal with respect to `device`:

    For most applications involving feedforward nets, you don't really have to
    bother with the `get_device` function (you don't even need to import it) or
    `device`.  The reason is that, in this version of DUlib, the `train` function
    silently takes care of moving things to the best device.

    The point is that activities such as splitting data into training/testing sets,
    centering and normalizing, and gauging accuracy of a trained model are often
    just as well done on the cpu even if you have a gpu. Training is really the
    main activity that you want to do on a gpu &mdash; and, again, DUlib now takes
    care of that for you.

    Likely reasons that you would want to bother with setting `device` are:
    * in the presence of multiple gpus, you want to specifying exactly which gpu you train on.
    * in the presence of gpu(s) you, for whatever reason, want to over-ride the use of a gpu and
      train on the cpu.
    * you want to non-trivially subclass DUlib's `TrainParams_` class.
    * you are creating something interesting using PyTorch's
      [optim](https://pytorch.org/docs/stable/optim.html) package.
    * you are computing r-squared in the presence of high-dimensional data.
    * you are using/building a recurrent net class.
  * The API, particularly in `du.lib`, is stabilizing quickly, as of this release, so it is
    worth it to upgrade. However, in transitioning to version 0.8, you need some
    refactoring. For example:
    * Calling `train` now looks, for example, like this:
      ``` python
      model = train(
          model,
          criterion,
          train_data = (xss_train, yss_train),
          test_data = (xss_test, yss_test),
          learn_params = {'lr': 0.01, 'mo': 0.9},
          epochs = 10)
      ```
      Notice that `learn_params` (the learning parameters) is, in this example,
      a dictionary.  It can also be an instance of the class `LearnParams_` (which is new in version 0.8; do
      `pydoc3 du.examples` to see a simple demonstration of using this class);
      or, `learn_params` can be a prebuilt optimizer from the [optim](https://pytorch.org/docs/stable/optim.html)
      package. (See the documentation for more details: `pydoc3 du.lib.train`).
    * Another change in version 0.8 is that `optimize_ols` now outputs a dict instead of a tuple.
      Hence you can, for example, do
      ``` python
      model = train(
          model,
          criterion,
          train_data = (xss, yss),
          learn_params = optimize_ols(xss),
          epochs = 10)
      ```
  * All code in the [DL@DU projects](https://github.com/sj-simmons/deep-learning#dldu) has been
    refactored to version 0.8.

<a id="latest"></a>
####

**Version 0.9** (December 23rd, 2019)
  * DUlib now includes a custom documentation viewer. Now you can type at the
    command line `pd du.lib.train` and have nicer viewing experience that when
    typing `pydoc3 du.lib.train`.
  * The current module structure is show below, with the modules in blue and
    some of the more common functions in red.  To view the current module
    structure, and to see suggested usage and more, type `pd du` at the
    command line.

    <p align="center">
      <img height="550" src="images/screenshot1.png">
    </p>

  * The core library is `du.lib`.  To quickly see usage and synopsis of the
    functions and classes available in `du.lib`, type `pd du.lib` at the
    command line (and scroll down to peruse detailed usage).

    <p align="center">
      <img height="550" src="images/screenshot2.png">
    </p>
  * Some command-line programs are now included. Try typing `dulib_polyreg_anim`
    at your command line.  You should see an animation that, along the way, looks
    something like:

    <p align="center">
      <img height="400" src="images/polyreg.svg">
    </p>

  * (Optional) graphing during training now looks like:

    <p align="center">
      <img height="400" src="images/graph2.svg">
    </p>

    Note: this is training on MNIST data with batch-size 20 using
    `du.conv.models.ConvFFNet` (type `pd du.conv.examples` at your command-line
    for details).


---

#### Modifying DUlib

If you want to experiment with the code in the libraries, then you can clone
to a local repo on your machine:
``` bash
git clone https://github.com/sj-simmons/DUlib.git
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
pip install git+https://github.com/sj-simmons/DUlib.git
```
or just
``` bash
pip install -U DUlib
```

Please let Simmons know of any new functionality that you implement and wish
to add to DUlib. Or just fork, create/fix/modify, and issue a pull request.

---

#### Todo
* In the `optimize_ols` function in [lib.py](du/lib.py), improve robustness the
  largest/smallest eigenvalue computation. Currently, larger datasets can cause numerical problems.
  The smallest eigenvalue can be negative, for example.

  See e.g. Applied Numerical Linear Algebra by Demmel &mdash; the most relevant chapters of which
  are [Chapter 3](../papers/demmelsvd.pdf), [Chapter 4](../papers/eigenvalues.pdf),
  [Chapter 5](../papers/eigenvalues2.pdf), and, particularly,
  [Chapter 6](../papers/eigenvalues3.pdf).

* Try to extend the functionality of `optimize_ols` to include mini-batch. As a
  starting point see the end of this
  [distill article on momentum](https://distill.pub/2017/momentum/).

