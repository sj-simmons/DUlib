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

#### Installation

First, install the library on your system using `pip3`. To install, for example,
version 0.1, issue this command at the command prompt:
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
Suppose you clone or fork this repo and modify or add to the code, and you
want to install (directly from your local repo) the modified libraries to your
local machine:
``` bash
cd DUlib
pip3 install -e .
```
Now suppose that you want to revert back to Simmons' version 0.2:
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git@v0.2
```
