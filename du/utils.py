#!/usr/bin/env python3
"""various utilities

"""
# Todo:
#   - finish compiling the regexs
#     - compile these once when the package is installed.
#   - implement strip to remove all char escape sequences used
#     in the markdown and see just plain text
#     - this is done for backticks, so do it for the rest.
#   - use inspect or whatever to transform all docstring in
#     all modules in DUlib
#   - in argparse check that all stuff was passed, like with
#     serial, before setting up that option.
#   - you have to implement strip because none of the doctest-
#     ing will work unless you strip off the markdown first.
#   - Consider changing more of the bools in stand_args to some
#     thing similar to the setup for cm.

import time
import torch
import argparse
import re
import signal
import inspect
import math

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

#class Graph:
#  """Dynamically graph progress during training.
#
#  Coming Soon
#  """
#  def _init__(self):
#    """Constructor
#
#    """
#    pass
#
#  def update(self):
#    pass

def standard_args(desc = '', epilog = '', **kwargs):
  """Factory for standard command line switches.

  Painlessly setup argparse command line switches for common
  hyperparameters, and return the parser object in such a way
  that the calling program can add more switches, if desired.

  !Notes!

  This function does not implement, for example, `bs` being set
  to -1 leading to (full) batch gradient descent. That should
  be implemented elsewhere (in `DUlib` it is implemented in the
  `du.lib.train` function). The `stand_args` function is handy only
  for eliminating boilerplate code when adding to one's program
  command-line switches for common hyper-parameters like learn-
  ing rate, momentum, etc.

  !Simple usage!

  Suppose that you want your program to have commandline op-
  tions for the learning rate, the momentum, and the no. of ep-
  ochs to train; and further that you want to provide yourself
  or some other user of your program with default values for
  the learning rate, and momentum, but that you want to require
  the user to specify the number of epochs on the command line.
  Additionally suppose that you want to add less commonly occu-
  ring commandline switch called 'foo', 'bar', and 'baz'. Then,
  in your program, put, for example:

      `import du.utils`

      `parser = du.utils.stand_args(`
          `'a short description of your program'`,
          `lr = 0.1`, `mo = 0.9`, `epochs = True`)
      `parser.add_argument('-foo'`,
          `help = 'a new thing that's an int with default 17'`,
          `type = int`, `default = 17`)
      `parser.add_argument('-bar'`,
          `help = 'a float that's required'`,
          `type = float`, `required = True`)
      `parser.add_argument('-baz'`,
          `help = 'a bool that's False by default'`,
          `action = 'store_true'`)
      `args = parser.parse_args()`
      ...
      `print('the learning rate is', args.lr)`
      `if args.baz: print('foo is', args.foo)`

                    _____________________


  Args:
    $desc$ (`str`): A short description of what the calling program
        does. Default: `''`.
    $epilog$ (`str`): More notes. Default: `''`.

  Kwargs:
    $lr$ (`Union[bool,float]`): If `True`, then add a commandline
        switch for the learning rate, returned to the calling
        program via the return parser object, with name `'lr'`.
        If a `float`, then setup the commandline switch to use
        that float as the default for the switch `-lr`. Default:
        `False`.
    $mo$ (`Union[bool,float]`): If `True`, add a switch for momentum.
        If a `float`, set that as the default momentum. Default:
        `False`.
    $bs$ (`Union[bool,int]`): Similarly to above, set up batchsize
        as a commandline switch, and note in the help string
        for the resulting switch the typical behavior that -1
        leads to batch gradient descent. Default: `False`.
    $epochs$ (`Union[bool,int]`): As above for the number of epochs
        over which train. Default: `False`.
    $props$ (`Union[bool,Tuple[float]]`): Similar to above, but
        for the proportions on which to train, validate, and/or
        test. Default: `False`.
    $dropout$ ('Union[bool,Tuple[float]]'): As above, useful for
        controlling dropout in your model. Default: `False`.
    $gpu$ (`Union[bool,Tuple[int]]`): Add a `gpu` switch. with a note
        in the help string to the effect:   ~which gpu (int or~
        ~ints separated by whitespace) for training/validating;~
        ~-1 for last gpu found (or to use cpu if no gpu found);~
        ~-2 for cpu~. Default: `False`.
    $graph$ (`Union[bool,int]`): Whether to add a switch for show-
        ing a graph during training, with a note in help to the
        effect: ~graph losses during training; redraw after this~
        ~many epochs~. Put `True` to enable a `-graph` switch that
        simply toggles `args.graph` to `True`. Default: `False`.
    $widths$ (`Union[bool,List[int]]`): This switch can be used in
        to pass in, for example, the widths of hidden layers of
        a dense feedforward network; the help string is ~the hid~
        ~den layer widths (integers separated by white-space)~.
        Add this to your program like, for example,
          `parser = du.utils.stand_args(..., widths=(100,), ...)`
        to have a single hidden layer.
        Default: `False`.
    $channels$ (`Union[bool,List[int]]`): Similar to `widths`; useful
        for, for example, convolutional nets. Default: `False`.
    $verb$ (`Union[bool,int]`): Whether or not to have a verbosity
        commandline switch. Default: `False`.
    $pred$ (`bool`): Whether or not add a `pred` switch, which refers
        to whether the program should simply make a prediction
        and exit. If this is set to `True` then files can be pro-
        vided on the commandline after the `-pred` switch. The
        help string returned is: ~don't train, only predict (on~
        ~any filenames that are optionally listed)~. Default:
        `False`.
    $ser$ (`bool`): Whether to have a commandline switch to deter-
        mine whether to serialize in the calling program. If
        this is `True` then the returned `parser` object will be
        set up so that the switch `-ser` stores `True`. The help
        string returned is: ~toggle serializing the trained mo~
        ~del~. Default: `False`.
    $pre$ (`bool`): Whether to have a commandline switch to specify
        whether to load a pretrained model. If `True` the return-
        ed `parser` object will be set up so that the switch `-pre`
        stores `True`. The help string returned is ~toggle loading~
        ~pre-trained model~. Default: `False`.
    $cm$ (`Union[None,bool]`): Whether to have a command-line switch
        to determine whether to show the confusion matrix. If
        this is `True`(resp. `False`), then the returned parser ob-
        ject will be set up so that the switch `-cm` stores `False`
        (`True`). The help string returned is: ~toggle showing con~
        ~fusion matrix~. Default: `None`.
    $seed$ (`bool`): Same as `ser` but for `seed` which refers to the
        random seed. Default: `False`.
    $print_lines$ (`Union[bool,tuple[int]]`): Whether to add switch
        controlling compressed printing to console; help string
        is: ~ints separated by whitespace controlling the no. of~
        ~lines to print before/after the ellipsis; put -1 to di-~
        ~sable compressed printing~. A length one tuple is dupli-
        cated to a length two tuple. Default: `False`.

  Returns:
    (`argparse.ArgumentParser`). The parser object to which the
        calling program can add more names.
  """
  _check_kwargs(kwargs,['lr','mo','bs','epochs','seed','props','gpu',
      'graph','ser','pre','pred','widths','channels','verb','cm',
      'print_lines','dropout'])
  lr = kwargs.get('lr', False)
  mo = kwargs.get('mo', False)
  bs = kwargs.get('bs', False)
  epochs = kwargs.get('epochs', False)
  seed = kwargs.get('seed', False)
  cm = kwargs.get('cm', None)
  props = kwargs.get('props', False)
  gpu = kwargs.get('gpu', False)
  graph = kwargs.get('graph', False)
  ser = kwargs.get('ser', False)
  pre = kwargs.get('pre', False)
  pred = kwargs.get('pred', False)
  widths = kwargs.get('widths',False)
  channels = kwargs.get('channels',False)
  verb = kwargs.get('verb',False)
  print_lines = kwargs.get('print_lines',False)
  dropout = kwargs.get('dropout',False)

  parser = argparse.ArgumentParser(
      description = desc,
      epilog = epilog,
      formatter_class = argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument

  if isinstance(lr, float):
    parser.add_argument('-lr', type=float, help='learning rate', default=lr)
  elif lr:
    parser.add_argument('-lr', type=float, help='learning rate',required=True)

  if isinstance(mo, float):
    parser.add_argument('-mo', type=float, help='momentum', default=mo)
  elif mo:
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

  hstr='proportion(s) to train, and/or validate, and/or test on'
  if not isinstance(props, bool) and isinstance(props, tuple):
    parser.add_argument('-props', type=float, help=hstr,
        metavar='props', nargs='*', default=props)
  elif isinstance(props, bool) and props:
    parser.add_argument('-props', type=int, help=hstr, metavar='props',
        nargs='*', required=True)

  if not isinstance(dropout, bool) and isinstance(dropout, tuple):
    parser.add_argument('-dropout', type=float, help='dropout proportion',
        nargs='*', default=dropout)
  elif isinstance(dropout, bool) and dropout:
    parser.add_argument('-dropout', type=float, help='dropout proportion',
        required=True)

  hstr='which gpu (int or ints separated by whitespace) for\
      training/validating; -1 for last gpu found (or to use cpu\
      if no gpu found); -2 for cpu'
  if not isinstance(gpu, bool) and isinstance(gpu, tuple):
    parser.add_argument('-gpu', type=int, help=hstr, metavar='gpu',
        nargs='*', default=gpu)
  elif isinstance(gpu, bool) and gpu:
    parser.add_argument('-gpu', type=int, help=hstr, metavar='gpu',
        nargs='*', required=True)

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

  hstr="the hidden layer widths (integers separated by white-space)"
  if not isinstance(widths, bool) and\
      all(isinstance(item, int) for item in widths):
    parser.add_argument('-widths',type=int,help=hstr,metavar='widths',
        nargs='*',default=widths)
  elif isinstance(widths, bool) and widths:
    parser.add_argument('-widths',type=int,help=hstr,metavar='widths',
        nargs='*',require=True)

  hstr="the channels (integers separated by white-space)"
  if not isinstance(channels, bool) and\
      all(isinstance(item, int) for item in channels):
    parser.add_argument('-channels',type=int,help=hstr,metavar='channels',
        nargs='*',default=channels)
  elif isinstance(channels, bool) and channels:
    parser.add_argument('-channels',type=int,help=hstr,metavar='channels',
        nargs='*',require=True)

  if pred:
    hstr="don't train, only predict (on filenames that are optionally listed);"+\
        " if True, then don't use files, but args.pred is now True"
    parser.add_argument('-pred',help=hstr,type=str,metavar='example',nargs='*')

  if seed:
    hstr='toggle setting random seed'
    parser.add_argument('-seed', help=hstr, action='store_true')

  if ser:
    hstr='toggle serializing the trained model'
    parser.add_argument('-ser', help=hstr, action='store_true')

  if pre:
    hstr='toggle loading pre-trained model'
    parser.add_argument('-pre', help=hstr, action='store_true')

  if isinstance(cm, bool):
    hstr='toggle showing the confusion matrix'
    if cm:
      parser.add_argument('-cm', help=hstr, action='store_false')
    else:
      parser.add_argument('-cm', help=hstr, action='store_true')

  hstr='ints separated by whitespace controlling no. lines to print\
    before/after the ellipsis; put -1 to disable compressed printing'
  if not isinstance(print_lines, bool) and isinstance(print_lines, tuple):
    parser.add_argument('-print_lines', type=int, help=hstr,
        metavar='print_lines', nargs='*', default=print_lines)
  elif isinstance(print_lines,bool) and print_lines:
    parser.add_argument('-print_lines', type=int, help=hstr,
        metavar='print_lines', nargs='*', required=True)

  return  parser

def args2string(args, keys, timestamp=True, maxlength=100, color=True):
  """Return string with selected values from args namespace.

  E.g., if your `args.Namespace()` contains 'lr' and 'mo', and you
  want to include only info about 'lr' in the output string then
  call this like `args2string(args, ['lr'], False)`.

  Args:
    $args$ (`argparse.ArgumentParser`): Instance of `ArgumentParser`.
    $keys$ (`List[str]`). The keys of `args` to be included in the
        output string.
    $timestamp$ (`bool`): Whether to prepend the output string
        a timestamp. Default: `True`.
    $maxlength$ (`int`): Start wrapping the output string after
        this many characters. Default: `100`.
    $color$ (`bool`) Whether to highlight the string. Def.:`True`.

  Returns:
    `str`.

  >>> `parser = standard_args(epochs=10,lr=0.5,mo=0.9)`
  >>> `args = parser.parse_args()`
  >>> `args2string(args, ['lr'], timestamp=False, color=False)`
  'lr:0.5'
  """
  length = 0
  def add_info(base_str, info, length):
    if length +len(info) > maxlength:
      base_str += '\n'
      length = len(info)
    else:
      length += len(info)
    return base_str + info, length
  d = vars(args)
  string = time.ctime()+'\n' if timestamp else ''
  for kwarg in keys:
    assert kwarg in d.keys(), "{} is not in args.Namespace()".format(kwarg)
    if isinstance(d[kwarg],float):
      string, length = add_info(
          string, '~'+kwarg+'~'+':'+'`'+str(format_num(d[kwarg]))+'` ', length)
    else:
      string, length = add_info(
          string, '~'+kwarg+'~'+':'+'`'+str(d[kwarg]).replace(' ','')+'` ', length)
  return _markup(string[:-1], strip = not color)

def get_device(gpu = -1):
  """Get a device, among those available, on which to compute.

  Args:
    $gpu$ (`int`): The gpu to use. Set to -1 to use the last GPU
        found when GPUs are present; set this to -2 to override
        using a found GPU and instead use the (first) CPU. Defa-
        ult: `-1`.

  Returns:
    `torch.device`. An instance of `torch.device` which can then
        be passed to `torch` tensors and modules (using their `to`
        method).
  """
  if gpu > -2:
    return torch.device("cuda:{0}".format(
               (torch.cuda.device_count() + gpu) % torch.cuda.device_count()
           )) if torch.cuda.is_available() else torch.device('cpu:0')
  else:
    return torch.device('cpu:0')

def print_devices():
  """Print available devices.

  Returns:
    `str`.
  """

  print('number of available (CPU) threads:',torch.get_num_threads())
  if torch.cuda.is_available():
    try:
      import pycuda.driver as cuda
    except ImportError:
      print('Error: the pycuda package is not installed.')
    else:
      cuda.init()
      print("CUDA is available:")
      print("  ID of default device is:", torch.cuda.current_device())
      print("  Name of default device is:", cuda.Device(0).name())
      import pycuda.autoinit
      class aboutCudaDevices():
        def __init__(self):
          pass

        def num_devices(self):
          """Return number of devices connected."""
          return cuda.Device.count()

        def devices(self):
          """Get info on all devices connected."""
          num = cuda.Device.count()
          print("%d device(s) found:"%num)
          for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)"%i)

        def mem_info(self):
          """Get available and total memory of all devices."""
          available, total = cuda.mem_get_info()
          print("Available: %.2f GB\nTotal:     %.2f GB"\
              %(available/1e9,total/1e9))

        def attributes(self, device_id=0):
          """Get attributes of device with device Id = device_id"""
          return cuda.Device(device_id).get_attributes()

        def __repr__(self):
          """Class representation gives number of devices connected and
          basic info about them.
          """
          num = cuda.Device.count()
          string = ""
          string += ("%d CUDA device(s) found:\n"%num)
          for i in range(num):
            string += ("  %d) %s (Id: %d)\n"%((i+1),cuda.Device(i).name(),i))
            string +=\
                ("     Memory: %.2f GB\n"%(cuda.Device(i).total_memory()/1e9))
          return string
      print(aboutCudaDevices())
  else:
      print('The pycuda package is not installed.')

def format_num(number, digits=3):
  """Format a small or a large number nicely.

  Args:
    $number$ (`float`): A number.

  Returns:
    `str`.

  >>> `print(format_num(.00000006))`
  6e-08
  >>> `print(format_num(12332314123))`
  12.3B
  >>> `print(format_num(999999))`
  1M
  """
  if number < .005:
    string = f'{number:.{digits}g}'
  elif number < 0:
    string = f'{number:.{digits+1}g}'
  elif number > 999:
    number = float('{:.3g}'.format(number))
    mag = 0
    while abs(number) >= 1000 and mag < 4:
        mag += 1
        number /= 1000.0
    string='{}{}'.format('{:f}'.format(number).rstrip('0').rstrip('.'),['','K','M','B','T'][mag])
  else:
    string = f'{number:.3g}'
  return string

def _check_kwargs(passed, valid_keywords):
  """Check that kwargs are valid.

  Check that each string in `passed` is in `valid_keywords` and
  notify of problems.

  Args:
    $passed$ (`List[str]`): In practice, the keywords that were
        passed to the function, class, method, etc. from which
        `_check_kwargs` was called.
    $valid_keywords$ (`List[str]`): The valid keywords for said func-
        tion, class, method, etc.
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
  """Catch keyboard interrupt.

  Catch a user's <CONTROL>-C and try to break out of a `for` or
  a `while` loop.
  """
  def keyboardInterruptHandler(signal, frame):
    global interrupted
    interrupted = True
  signal.signal(signal.SIGINT, keyboardInterruptHandler)

class _Markdown:
  """Markdown for bold, underline, and a few colors.

  Here is the markdown

    `bold`, !underline somthing!, $make red$, ~blue~, |cyan|

  Don't try to use across line breaks. (Do that by hand)

  Also:
      ... become grey (kind of)
      seperators like ______________ get underlined, too
  """
  def __init__(self):
    #self.PURPLE = '\033[95m'
    #self.CYAN = '\033[96m'
    #self.DARKCYAN = '\033[36m'
    #self.BLUE = '\033[94m'
    #self.GREEN = '\033[92m'
    #self.YELLOW = '\033[93m'
    #self.RED = '\033[91m'
    #self.BOLD = '\033[1m'
    #self.UNDERLINE = '\033[4m'
    #self.END = '\033[0m'
    self.GREY = '\033[30m'
    self.RED = '\033[31m'
    self.GREEN = '\033[32m'
    self.YELLOW = '\033[33m'
    self.BLUE = '\033[34m'
    self.MAGENTA = '\033[35m'
    self.CYAN = '\033[36m'
    self.WHITE = '\033[37m'
    self.BOLD = '\033[1m'
    self.UNDERLINE = '\033[4m'
    self.END = '\033[0m'
    self.bold_pat = re.compile(r"`([^`]+)`")
    self.red_pat = re.compile(r"\$([^\$]+)\$")
    self.cyan_pat = re.compile(r"~([^~]+)~")
    #self.blue_pat = re.compile()
    #self.ellipses_pat = re.compile()
    #self.underline_pat = re.compile()
    #self.seperator_pat = re.compile()

  def bold(self, docstring, strip=False):
    """Makes bold words or phrases on one line surrounded by ` symbols."""
    if not strip:
      return re.sub(self.bold_pat, self.BOLD+r"\1"+ self.END, docstring)
    else:
      return docstring.translate(docstring.maketrans(dict.fromkeys('`')))

  def red(self, docstring, strip=False):
    """Makes red words or phrases on one line surrounded by $ symbols."""
    if not strip:
      return re.sub(self.red_pat, self.BOLD+self.RED+r"\1"+ self.END, docstring)
    else:
      return docstring.translate(docstring.maketrans(dict.fromkeys('$')))

  def cyan(self, docstring, strip=False):
    """Makes blue words or phrases on one line surrounded by ~ symbols."""
    if not strip:
      return re.sub(self.cyan_pat, self.CYAN+r"\1"+ self.END, docstring)
    else:
      return docstring.translate(docstring.maketrans(dict.fromkeys('~')))

  def blue(self, docstring, strip=False):
    """Makes cyan words or phrases on one line surrounded by | symbols."""
    if not strip:
      return re.sub(
          r'\|([^\|]+)\|', self.BOLD+self.BLUE+r'\1'+ self.END, docstring)
    else:
      return docstring.translate(docstring.maketrans(dict.fromkeys('|')))

  def ellipses(self, docstring, strip=False):
    """colors >>> and ... ."""
    if not strip:
      return re.sub(r'(>>>|\.\.\.)', self.MAGENTA+r'\1'+ self.END, docstring)
    else:
      return docstring

  def underline(self, docstring, strip=False):
    """Underlines words or phrases on one line surrounded by ! symbols."""
    s = re.sub(r'!([a-zA-Z0-9][a-zA-Z0-9 ,:\-]+)!',\
        self.UNDERLINE+r'\1'+self.END,docstring)
    return re.sub(r'\_\_\_([\_]+)',self.UNDERLINE+r'___\1'+self.END,s)

def _markup(docstring, md_mappings = _Markdown(), strip = False):
  """Process markdown.

  Looks at markdown and wraps with the appropriate char es-
  cape sequences, or strips away the markdown.

  Args:
    docstring (str):The docstring to process.
    md_mappings (class): An instance of a class all of
        whose methods map str -> str by default.
    strip (bool): Whether or not to remove the markdown.

  >>> _markup('$Loren$ _lipsum_ |dolor|') # doctest: +SKIP
  >>> print(_markup('$blah$',strip=True))
  blah
  >>> print(_markup('|blab|',strip=True))
  blab

  """
  for _, f in inspect.getmembers(md_mappings, inspect.ismethod)[1:]:
    if docstring is not None:
      docstring = f(docstring, strip = strip)
  return docstring

if __name__ == '__main__':
  import doctest

  # find the user defined functions
  _local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
       if callable(ob) and ob.__module__ == __name__]

  #remove markdown
  #  from the docstring for this module
  globals()['__doc__'] = _markup(globals()['__doc__'],strip = True)
  #  from the functions (methods are fns in Python3) defined in this module
  for _, _ob in _local_functions:
    if inspect.isfunction(_ob):
      _ob.__doc__ = _markup(_ob.__doc__,strip = True)
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
        _ob.__dict__[name].__doc__ = _markup(_meth.__doc__,strip = True)

  # run doctests
  failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)

  # print signatures
  if failures == 0:
    from inspect import signature
    for name, ob in _local_functions:
      print(name,'\n  ', inspect.signature(ob))
