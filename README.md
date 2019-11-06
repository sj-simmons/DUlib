<p align="right"> <b> DL@DU </b> </p> <a id="dldu"></a>

### DUlib
---

#### Installation

First, install the library on your system with `pip3`:
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git
```
(You might need to add the `--user` option to the above command.)
Then, in your program, do:
``` python
from du.lib import center, normalize, train, confusion_matrix
```
and/or
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
git clone https://github.com/sj-simmons/DUlib.git
```
Suppose you modify or add to the code, and you want to install (directly from
your local repo) the new libraries
to your machine:
``` bash
cd DUlib
pip3 install -e .
```
Now suppose that you want to revert back to Simmons' version:
``` bash
pip3 install git+https://github.com/sj-simmons/DUlib.git
```


