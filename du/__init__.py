__doc__ =\
"""
tools from `The DL@DU Project`.

The |module structure|, along with the (non-helper) $functions$
and $classes$, are, as of this release,

   |du|
   ├── |lib.py|  !the core library!
   │   ├── $center$         mean-center some data, and/or
   │   ├── $normalize$      normalize it
   │   ├── $coh_split$      split out testing data
   │   ├── $train$          train a model
   │   ├── $cross_validate$ cross-validate train a model
   │   ├── $confusion_matrix$
   │   ├── $r-squared$
   │   ├── $optimize_ols$   find optimal lr and momentum
   │   └── $LearnParams$    adaptive training hyper-params
   ├── |models.py|  !model classes for feed-forward nets!
   │   └── $DenseFF$        factory for dense FF nets
   ├── |examples.py|
   ├── |conv|  !convolutional tools for images and more!
   │   └── |models.py|
   ├── |rec|   !recurrence and natural language processing!
   │   ├── |lib.py|
   │   │   ├── |ngrams.py|  model n-grams given a word corpus
   │   │   └── |cbow.py|    model CBOW word vecs for a corpus
   │   ├── |examples.py|
   │   └── |models.py|
   └── |util.py|  !potentially helpful utilites!
       └── $stand_args$     easily set up commandline options

Suggestions for importing `DUlib`'s functionality into your
programs:
  ...
  `import du.lib as dulib`
  `from du.models import DenseFF`
  ...

Then, in your code, do, say,
  ...
  `xss, _ = dulib.center(xss)`
  ...
  `model = DenseFF(`
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

Our goal in writing this library is two-fold. We wish to
provide well-designed, high-quality tools that package the
power and extensibility of `PyTorch`, and with which we can
create our experiments and designs in code in near real-time.

We also hope to provide a concise vehicle by which we may
hone our command of Machine and Deep Learning and explore
the nature of the way, in fact, not machines, but humans,
wield such tools as they unearth the secrets held by data.

The true power of Deep Learning which, since it is formul-
ted in terms of neural nets, was once thought to be rather
prosiac (particularly by mathematicians), lies in giving
machines near absolute freedom to discover patterns in data.

This freedom combined with the enormity of data available
in the modern information era, has fueled the resurgence in
viability of neural networks.

None of this would be possible without artfully efficient
implementations of the massive ~chain rule~ computations
driving the convergence of deep networks. Those algorithms
comprise ~gradient descent~, the ~stochastic~ version of which
is the workhorse of modern machine learning.

As late as the 90s, some doubted the computational infeas-
ibility of deep nets. Yet, not only do the most multilay-
ered of networks (and even ~recurrent~ networks) converge,
they produce windows into dimensions of big data previous-
ly unseen. Cutting-edge techniques in ML and DL place mach-
ines at liberty to even conceptualize parts of their own
architectures.

!Quick Start!

A good place to begin building your expertise is by reading
the documentation in |examples.py|, which you do by issuing

    |pydoc3 du.examples|

at the command line.

In fact, our goal is to provide demonstrations for most
all of the over-arching functionality herein. Hence, exam-
ples can also be found in the submodules of DUlib. Type,
e.g.,

    |pydoc3 du.conv.examples|

or

    |pydoc3 du.rec.examples|

to see examples of involving, respectively convolution-
al and recurrent nets.
                 _____________________

Many of the demonstrations in DUlib come with visualiza-
tions (assuming that you have matplotlib installed). All
of these can be run from the command line.

For example you can run an animation of gradient descent
with the command

    |dulib_linreg_anim|

or, to see gradient descent in action in a more general
setting,

    |dulib_polyreg_anim|

Try typing at your command line

    |dulib|<TAB><TAB>

which means type the letters `dulib` and then hit the `TAB`
key twice. You will see all of the available visualizations.

Alternatively, just start reading the docs, where the al-
gorithms behind the visualizations are discussed at length.
                  _____________________

A technical note on

          $breaking out of charset encoding hell$

If you can easily and normally read the phrase above, then
you are not even in charset encoding hell. In this case, you
can simply

         $Go ahead and get started Deep Learning!$

However if that line (and, in fact, some other words/phras-
es in the discussion above) is enclosed in weird boxes, then
you need to break out of so-called charset encoding hell.
There are a number of escape routes around such rendering
issues; here are two:

1) There is a good chance that the problems your are exper-
   ienc- ing are due to your being in IDLE (which is the de-
   fault IDE that ships with Python) or some other IDE that
   doesn't play nice with ansi escape sequences.

   Recommendation: simple consume the documentation for this
   library by using `pydoc3` on the command line. That way you
   you can enjoy a few enhancement like bolding and
   coloring of certain words. This helps with quickly and
   easily finding the information you need when building on
   this library.

2) Alternatively, you can strip out all of the escape seq-
   uences that causing problems for you by simply ...

"""
from du.util import _markup

__author__ = 'Scott Simmons'
__status__ = 'Development'
__date__ = '12/06/19'
__version__ = '0.8.5'
__doc__ = _markup(__doc__)
