#!/usr/bin/env python3
"""Utilities
"""
# Todo:
#   - finish compiling the regexs
#     - compile these once when the package is installed.
#   - implement strip to remove all char escape sequences used
#     in the markdown and see just plain text
#   - use inspect or whatever to transform all docstring in
#     all modules in DUlib
#   - in argparse check that all stuff was passed, like with
#     serial, before setting up that option.

import torch
import argparse
import re
import signal
import inspect
import du.util

__author__ = 'Scott Simmons'
__version__ = '0.8.5'
__status__ = 'Development'
__date__ = '12/06/19'

def stand_args(desc = '', **kwargs):
  '''Factory for command line switches.

  Painlessly, setup argparse switches, etc. for common hyper-
  parameters, and return the parser object in such a way that
  the calling program can add more switches, if desired.

  Notes:

  This function does not implement, for example, `bs` being set
  to -1 leading to (full) batch gradient descent. That is imp-
  lemented elsewhere (in the `du.lib.train` function). This
  function is handy solely for eliminating boilerplate code
  when adding to one's program commandline switches for common
  hyper-parameters like learning rate, momentum, etc.

  Simple usage:

  Suppose that you want your program to have commandline op-
  tions for the learning rate, the momentum, and number of
  epochs to train; and further that you want to provide your-
  self or some other user of your program with default values
  for the learning rate, and momentum, but that you want to
  require the user to specify the number of epochs on the com-
  mand line. Additionally suppose that you want to add a less
  common commandline switch called 'foo'.

  Then, in your program, put

    `import du.util`

    `parser = du.util.stand_args(`
        `'a short description of your program'`,
        `lr = 0.1`,
        `mo = 0.9`,
        `epochs = True`)
    `parser.add_argument('-foo',help='some new switch')`
    `args = parser.parse_args()`
    ...
    `print('the learning rate is', args.lr)`

  Args:
    desc (str): A short description of what the calling pro-
        gram does.  Default: ''.

  Kwargs:
    lr (Union[bool,float]): If `True`, then add a commandline
        switch for the learning rate, returned to the calling
        program via the return parser object with name `'lr'`.
        If a `float`, then setup the commandline switch to
        use that float as the default `lr`. Default: False.
    mo (Union[bool,float]): If `True`, add a switch for mom-
        entum.  If a `float`, set that as the default momen-
        tum. Default: False.
    bs (Union[bool,int]): Similarly to above, set up batchsize
        as a commandline switch, and note in the description
        the typical behavior that -1 leads to batch gradient
        descent. Default: False.
    epochs (Union[bool,int]): As above for number of epochs
        over which train. Default: False.
    prop (Union[bool,float]): As above, but for the propor-
        tion on which train. Default: False.
    gpu (Union[bool,int]): Add a `gpu` switch. with a note
        in the description to the effect: which gpu, if more
        than one is found; -1 for last gpu found; -2 for cpu
        Default: False.
    graph (Union[bool,int]): Whether to add commandline switch
        for showing a graph during training, with a note in the
        description to the effect: graph losses during train-
        ing; redraw after this many epochs. Just put `True` to
        enable a `-graph` switch that simply toggles on graph-
        ing. Default: False.
    widths (Union[bool,List[int]]): This switch is used in
        DUlib demos to pass in, for example, the number and
        widths of the layers of a dense feed-forward network.
        Default: False.
    verb (Union[bool,int]): Whether or not to have a verbosity
        commandline switch. Default: False.
    pred (bool): Whether or not add a `pred` switch, which re-
        fers to whether the program should simply make a pred-
        iction and exit. If this is True then files can provid-
        ed on the command line after the `-pred` switch. The
        help string returned is: don't train, only predict (on
        any filenames that are optionally listed).
        Default: False.
    ser (bool): Whether to have a commandline switch to deter-
        whether to serialize in the calling program. If this
        is true then the returned `parser` will be set up so
        that the switch `-ser` is stores `True`. The help
        string returned is: toggle setting random seed. See
        above for an example.
        Default: False.
    seed (bool): Same as `ser` but for `seed` which refers to
        the random seed. Default: False.

  Returns:
    (argparse.ArgumentParser). The parser object to which the
        calling program can add more names.
  '''
  du.util._check_kwargs(kwargs,['lr','mo','bs','epochs','seed','prop', 'gpu',\
      'graph','ser','pred','widths','verb'])
  lr = kwargs.get('lr', False)
  mo = kwargs.get('mo', False)
  bs = kwargs.get('bs', False)
  epochs = kwargs.get('epochs', False)
  seed = kwargs.get('seed', False)
  prop = kwargs.get('prop', False)
  gpu = kwargs.get('gpu', False)
  graph = kwargs.get('graph', False)
  ser = kwargs.get('ser', False)
  pred = kwargs.get('pred', False)
  widths = kwargs.get('widths',False)
  verb = kwargs.get('verb',False)

  parser = argparse.ArgumentParser( description = desc, formatter_class =\
      argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument

  if isinstance(lr, float):
    parser.add_argument('-lr', type=float, help='learning rate', default=lr)
  elif lr:
    parser.add_argument('-lr', type=float, help='learning rate',required=True)

  if isinstance(mo, float):
    parser.add_argument('-mo', type=float, help='momentum', default=mo)
  elif lr:
    parser.add_argument('-mo', type=float, help='momentum',required=True)

  hstr='the mini-batch size; set to -1 for (full) batch gradient descent'
  if not isinstance(bs, bool) and isinstance(bs, int):
    parser.add_argument('-bs', type=int, help=hstr, default=bs)
  elif isinstance(bs,bool) and bs:
    parser.add_argument('-bs', type=int, help=hstr, required=True)

  hstr='number of epochs to train'
  if not isinstance(epochs, bool) and isinstance(epochs, int):
    parser.add_argument('-epochs', type=int, help=hstr, default=epochs)
  elif isinstance(epochs,bool) and epochs:
    parser.add_argument('-epochs', type=int, help=hstr, required=True)

  hstr='proportion to train on'
  if isinstance(prop, float):
    parser.add_argument('-prop', type=float, help=hstr, default=prop)
  elif prop:
    parser.add_argument('-prop', type=float, help=hstr, required = True)

  hstr='which gpu, if more than one is found; -1 for last gpu found; -2 for cpu'
  if not isinstance(gpu, bool) and isinstance(gpu, int):
    parser.add_argument('-gpu', type=int, help=hstr, default=gpu)
  elif isinstance(gpu, bool) and gpu:
    parser.add_argument('-gpu', type=int, help=hstr, required=True)

  if not isinstance(graph, bool) and isinstance(graph, int):
    hstr='1 to graph losses while training; > 1 to redraw after that many epochs'
    parser.add_argument('-graph', help=hstr, type=int, default=graph)
  elif isinstance(graph, bool) and graph:
    hstr='toggle graphing'
    parser.add_argument('-graph', help=hstr, action='store_true')

  if not isinstance(verb, bool) and isinstance(verb, int):
    parser.add_argument('-verb', type=int, help='verbosity', default=verb)
  elif isinstance(verb, bool) and verb:
    parser.add_argument('-verb', type=int, help='verbosity', required=True)

  hstr="the hidden layer widths (integers separated by white-space"
  if not isinstance(widths, bool) and\
      all(isinstance(item, int) for item in widths):
    parser.add_argument('-widths',type=int,help=hstr,metavar='widths',
        nargs='*',default=widths)
  elif isinstance(widths, bool) and widths:
    parser.add_argument('-widths',type=int,help=hstr,metavar='widths',
        nargs='*',require=True)

  if pred:
    hstr="don't train, only predict (on filenames that are optionally listed)"+\
        "if `True` don't use files but `args.pred` is now `True` "
    parser.add_argument('-pred',help=hstr,type=str,metavar='example',nargs='*')

  if seed:
    hstr='toggle setting random seed'
    parser.add_argument('-seed', help=hstr, action='store_true')

  if ser:
    hstr='toggle serialize the trained model'
    parser.add_argument('-ser', help=hstr, action='store_true')

  return  parser

def get_device(gpu = -1):
  """Get the best device to run on.

  Args:
    gpu (int): The gpu to use. Set to -1 to use the last gpu
        found when gpus are present; set to -2 to override
        using a found gpu and use the cpu. Default -1.

  Returns:
    str. A string that can be passed using the `to` method
        of `Torch` tensors and modules.
  """
  if gpu > -2:
    return torch.device( "cuda:{0}".format(
               (torch.cuda.device_count() + gpu) % torch.cuda.device_count()
           )) if torch.cuda.is_available() else "cpu"
  else:
    return 'cpu'

def format_num(number):
  '''Format a small number nicely.

  Args:
    number (float): A number.

  Returns:
    str.
  '''
  if number < .005: string = '{:.4g}'.format(number)
  else: string = '{:.5g}'.format(number)
  return string

def _parse_data(data_tuple, device = 'cpu'):
  '''Helper function for the train function.

  Args:
    data_tuple Tuple[tensor]: Length either 2 or 3.

  Returns:
    Tuple[tensor].
  '''
  feats = data_tuple[0].to(device); targs = data_tuple[-1].to(device)
  if len(data_tuple) == 3:
    feats_lengths = data_tuple[1].to(device)
    assert len(feats_lengths) == len(feats),\
        "No. of feats lengths ({}) must equal no. of feats ({}).".\
            format(len(feats_lengths), len(feats))
  else:
    assert len(data_tuple) == 2, 'data_tuple must have len 2 or 3'
    feats_lengths = None
  assert len(feats) == len(targs),\
      "Number of features ({}) must equal number of targets ({}).".\
          format(len(feats), len(targs))
  return feats, feats_lengths, targs

def _check_kwargs(passed, valid_keywords):
  """ Check that each string in passed is in valid and notify
  of problems.

  Args:
    passed (List[str]): In practice, the keywords that were
        passed to the function, class, method, etc. from which
        `_check_kwargs` was called.
    valid_keywords (List[str]): The valid keywords for said
        function, class, method, etc.
  """
  for keyword in passed:
    assert keyword in valid_keywords,\
        '{} is not a valid argument keyword'.format(keyword)

def _catch_sigint():
  """Catch keyboard interrupt signal."""
  def keyboardInterruptHandler(signal, frame):
    #print("KeyboardInterrupt (ID: {}) caught. Cleaning up...".format(signal))
    print("\n")
    exit(0)
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

def _catch_sigint_and_break():
  """Catch keyboard interrupt signal and break out of a `for` or
  `while` loop.
  """
  def keyboardInterruptHandler(signal, frame):
    global interrupted
    interrupted = True
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

def _catch_wrong_tensors(crit, xss, yss):
  '''Try to catch when someone passes the wrong size or type.
  '''
  pass

class _Color:
  """Mini markdown.

  Here is the markdown
    `bold`, !underline somthing!, $make red$, ~blue~, |cyan|

  Don't try to use across line breaks. (Do that by hand)

  Also:
      ... become grey (kind of)
      seperators like ______________ get underlined too
  """
  def __init__(self):
    self.PURPLE = '\033[95m'
    self.CYAN = '\033[96m'
    self.DARKCYAN = '\033[36m'
    self.BLUE = '\033[94m'
    self.GREEN = '\033[92m'
    self.YELLOW = '\033[93m'
    self.RED = '\033[91m'
    self.BOLD = '\033[1m'
    self.UNDERLINE = '\033[4m'
    self.END = '\033[0m'
    self.bold_pat = re.compile(r"`([^`]+)`")
    self.red_pat = re.compile(r"\$([^\$]+)\$")
    self.blue_pat = re.compile(r"~([^~]+)~")
    #self.cyan_pat = re.compile()
    #self.gray_pat = re.compile()
    #self.underline_pat = re.compile()
    #self.seperator_pat = re.compile()

  def bold(self, docstring, strip=False):
    """Makes bold words or phrases on one line surrounded by ` symbols."""
    return re.sub(self.bold_pat, self.BOLD+r'\1'+ self.END, docstring)

  def red(self, docstring, strip=False):
    """Makes red words or phrases on one line surrounded by $ symbols."""
    return re.sub(self.red_pat, self.BOLD+self.RED+r'\1'+ self.END, docstring)

  def blue(self, docstring, strip=False):
    """Makes blue words or phrases on one line surrounded by ~ symbols."""
    return re.sub(self.blue_pat , self.BOLD+self.BLUE+r'\1'+ self.END,docstring)

  def cyan(self, docstring, strip=False):
    """Makes cyan words or phrases on one line surrounded by | symbols."""
    return re.sub(
        r'\|([^\|]+)\|', self.DARKCYAN+r'\1'+ self.END, docstring)

  def gray(self, docstring, strip=False):
    """Makes gray >>> and ... ."""
    return re.sub(r'(>>>|\.\.\.)', self.BLUE+r'\1'+ self.END, docstring)

  def underline(self, docstring, strip=False):
    """Underlines words or phrases on one line surrounded by _ symbols."""
    s = re.sub(r'!([a-zA-Z][a-zA-Z \-]+)!',\
        self.UNDERLINE+r'\1'+self.END,docstring)
    return re.sub(r'[\_\_\_]([\_]+)',self.UNDERLINE+r'___\1'+self.END,s)

#def _shorten_doc(docstring):
#  """Strip out un-useful info from a docstring of a subclass about
#  attributes etc from the base class.
#
#  docstring = "blah blab stuff\n"+'-'*70+"\n"+"Methods inherited from"
#  docstring = "blah blab stuff"+"Methods inherited from"
#  shorten_doc(docstring)
#  -----------------------
#  """
#  return docstring.split('-'*70)[0]

def _markup(docstring, md_mappings = _Color(), strip = False):
  """Process markdown.

  Looks at markdown and wraps with the appropriate char
  escape sequences, or strips away the markdown.

  Args:
    docstring (str): The docstring to process.
    md_mappings (class): An instance of a class all of
        whose methods map str -> str by default.
    strip (bool):  Whether or not to remove the markdown.

  >>> _ansi('$Loren$ _lipsum_ |dolor|') # doctest: +SKIP
  '\x1b[1m\x1b[91mLoren\x1b[0m \x1b[4mlipsum\x1b[0m \x1b[36mdolor\x1b[0m'
  """
  for _, f in inspect.getmembers(md_mappings, inspect.ismethod)[1:]:
    docstring = f(docstring)
  return docstring

if __name__ == '__main__':
  import doctest
  failures, _ = doctest.testmod()

  if failures == 0:
    # Below prints only the signature of locally defined functions.
    from inspect import signature
    local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
        if callable(ob) and ob.__module__ == __name__]
    for name, ob in local_functions:
      print(name,'\n  ',signature(ob))

#globals()['__doc__']=_markup(globals()['__doc__'])

#stand_args.__doc__=_markup(stand_args.__doc__)


