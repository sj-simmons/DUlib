<p align="right"> <b> DL@DU </b> </p> <a id="dldu"></a>

### DUlib
---

#### Versions

**0.1** (March 2019)
  * If you were part of the DL@DU project before about November 1, 2019 and
    you used any of libraries in the old repo then you can install this version
    and have all your code work as written with the sole exception of modifying
    the way you import functions from library (see below).
  * Quick install: `pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.1`

**0.2** (November 6, 2019)
  * added [wordlib.py](du/wordlib.py) for NLP applications.
  * the functionality and functions (like `train`) in the original library
    have been enhanced and tweaked.  Some refactoring of your code might be
    required to transition from version 0.1 to 0.2.
  * Quick install: `pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.2`

---

#### Documentation

Perusing this repo will only show you the code for the most recent release
version (unless you try to dig through previous commits).  So if you are interested
in the functionality of say version 0.1, Then don't click on the code on this web
page; instead you can do this.
* Install version 0.1 on your systems as above.
* Then type `pydoc3 du` at the command line.  You'll see something like this:
  ``` bash
  Help on package du:

  NAME
      du

  PACKAGE CONTENTS
      lib
      wordlib

  VERSION
      0.2

  FILE
      /home/ssimmons1331/Insync/Code/DUlib/du/__init__.py
  ```
  which tells you what modules the `du` package that you have installed contains.
  Note that Simmons has version 0.2 installed which consists of 2 modules: one called
  `lib` and the other called `wordlib`.

  Then you can type, for example:
  ``` bash
  pydoc3 du.lib
  ```
  and see all the documentation for `du.lib` (which is where basic functions like
  `train` live).  Similarly you can type `pydoc3 du.wordlib` if you have version 0.2
  installed.
---

#### Installation

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
or, say,
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
Suppose you clone to a local and modify or add to the code in the libraries, and
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
