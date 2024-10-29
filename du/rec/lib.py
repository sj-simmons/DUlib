#!/usr/bin/env python3
'''NLP utilities.

dir_to_lines
   (dir_name:str, small:int=-1) -> Dict
file_to_lines
   (file_name:str) -> List
invertListDict
   (d:Dict) -> Dict
line_to_chars
   (line:str) -> List
line_to_ngrams
   (line:str, n:int) -> List
line_to_tokens
   (line:str) -> List
lines_to_tokens
   (lst:List) -> List
make_ngram
   (tokens:List) -> Tuple
make_ngrams
   (lines:List, n:int) -> List
make_token2index
   (tokens:Union, pad:str=None) -> Dict
ngrams_to_tensor
   (ngrams:List, token2index:Dict) -> Tuple
pad_sequences
   (seqs:List) -> List
pca_plot
   (tokens, vects)
tokenize
   (word:str) -> str
tsne_plot
   (words, vects, num_components=3)
unpad_sequence
   (seq:List, padding_idx:int=0) -> List
'''
import glob
import io
import os
from string import punctuation
from typing import List, Dict, Tuple, Sequence, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import du.utils

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

# define some type aliases
Ngram = Tuple[List[str], str]
Tensor = torch.Tensor
LongTensor = torch.LongTensor

def file_to_lines(file_name: str) -> List[str]:
  '''Read in lines of a file to a list.

  Args:
    file_name (list): The name of the file to read.

  Returns:
    List[str]. A list whose items are the lines of the file
        that was read..
  '''
  with open(file_name) as f:
    corpus = f.read()
  return [line for line in corpus.split('\n') if line != '']

def dir_to_lines(dir_name: str, small: int = -1) -> Dict[str, List]:
  '''For each file (with extension '.txt') in a directory,
  read its lines into a list.

  Args:
    small (int): Read only this many lines from each file.

  Returns:
    Dict[str, List[str]]. A dict of the lists indexed by the
        filename bases with values the the lines in that file
        in the form of a list of strings..
  '''
  d = {}
  for filename in glob.glob(dir_name+'/*.txt'):
    basename = os.path.splitext(os.path.basename(filename))[0]
    d[basename.lower()] = file_to_lines(filename)[:small]
  return d

def invertListDict(d: Dict[Any, List[Any]]) -> Dict[Any, Any]:
  '''Invert a dict whose values are lists.

  Args:
    d (Dict[Any, List[Any]]). The dict to be inverted.

  Returns:
    Dict[Any, Any]: The dict whose keys are the elements of
    lists that are values of `d` and whose values are the
    apropriate key.

  >>> d = {'one': [1.2, 1.34], 'two': [2.0, 2.5]}
  >>> d_inverted={1.2:'one', 1.34:'one', 2.0:'two', 2.5:'two'}
  >>> invertListDict(d) == d_inverted
  True
  '''
  return {item: k for k, lst in d.items() for item in lst}

def tokenize(word: str) -> str:
  '''Return the token for a word.

  Args:
    word (str): The word to tokenize.

  Returns:
    (str). The tokenized word.

  >>> tokenize("You're,")
  "you're"
  '''
  return word.strip(punctuation).lower()

def line_to_tokens(line: str) -> List[str]:
  '''Split line on whitespace and return the resulting list
  of tokens.

  Args:
    line (str):

  :rtype: List[str]

  >>> line_to_tokens('San ti wenti; Lang tu tong')
  ['san', 'ti', 'wenti', 'lang', 'tu', 'tong']
  >>> line_to_tokens('??')
  []
  '''
  tokens = [tokenize(s) for s in line.split()]
  return [token for token in tokens if token != '']

def lines_to_tokens(lst: List[str]) -> List[str]:
  return [line_to_tokens(item) for item in lst]

def make_token2index(tokens: Union[List[str],List[List[str]]],pad: str = None)\
    -> Dict[str, int]:
  '''Return a dict whose keys are the unique tokens and whose
  values are from the set {0,1,...,len(dict)}.

  Args:
    tokens (Union[List[str], List[List[str]]]): List or list
        of lists of tokens the elements of which are the
        tokens of the corpus.
    pad (str): Add this string to the corpus with index 0.

  Returns:
    Dict[str, int]. A dict (often called vocab) mapping tokens
        to indices.

  >>> make_token2index([['a', 'z', 'i'], ['r', 'z'], ['i']])
  ... # doctest: +SKIP
  '''
  if isinstance(tokens[0], list):
    tokens = [item for sublist in tokens for item in sublist]
  vocab = set(tokens)
  if pad:
    return {token: i for i, token in enumerate([pad] + sorted(vocab))}
  else:
    return {token: i for i, token in enumerate(sorted(vocab))}

def make_ngram(tokens: List[str]) -> Ngram:
  '''Return the ngram for a list of tokens.

  :type tokens: List[str]
  :rtype: Ngram = Tuple[List[str], str]

  >>> make_ngram(['one', 'giant', 'leap', 'for', 'mankind'])
  (['one', 'giant', 'leap', 'for'], 'mankind')
  '''
  return tokens[:-1], tokens[-1]

def line_to_chars(line: str) -> List[str]:
  '''Return list of characters in a line.

  Args:
    line (str): The line to be processed.

  Returns:
    List[str]: The list of (lowercased) characters.

  >>> line_to_chars('Coltrane')
  ['c', 'o', 'l', 't', 'r', 'a', 'n', 'e']
  >>> line_to_chars("L'Hopital")
  ['l', "'", 'h', 'o', 'p', 'i', 't', 'a', 'l']
  '''
  return list(line.lower())

def line_to_ngrams(line: str, n: int) -> List[Ngram]:
  '''Return a list of ngrams for line.

  :rtype: List[Ngram] = List[Tuple[List[str], str]]

  >>> line_to_ngrams('Yi er san si.', 3)
  [(['yi', 'er'], 'san'), (['er', 'san'], 'si')]
  >>> line_to_ngrams('Yi er san si.', 4)
  [(['yi', 'er', 'san'], 'si')]
  >>> line_to_ngrams('??', 4)
  []
  '''
  tokens = line_to_tokens(line)
  return [make_ngram(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def make_ngrams(lines: List[str], n: int) -> List[Ngram]:
  '''Return list of ngrams.

  :param lines: List whose elements are the lines of the corpus.
  :type lines: List[str]
  :rtype: List[Ngram] = List[Tuple[List[str], str]]

  >>> make_ngrams(['Veni vidi vici'], 2)
  [(['veni'], 'vidi'), (['vidi'], 'vici')]
  >>> make_ngrams(['When it comes', 'will it come in darkness'], 4)
  [(['will', 'it', 'come'], 'in'), (['it', 'come', 'in'], 'darkness')]
  '''
  line_wise_ngrams = [line_to_ngrams(line, n) for line in lines]
  return [ngram for line in line_wise_ngrams for ngram in line]

def pad_sequences(seqs: List[List[int]]) -> List[List[int]]:
  '''Zero pad sequences.

  >>> pad_sequences([[2, 3], [2, 1, 2, 7], [1]])
  ([[2, 3, 0, 0], [2, 1, 2, 7], [1, 0, 0, 0]], [2, 4, 1])
  '''
  lengths = [len(seq) for seq in seqs]
  max_length = max(lengths)
  new_seqs = [[0 for _ in range(max_length)] for _ in range(len(seqs))]
  for i, seq in enumerate(seqs):
    for j, entry in enumerate(seq):
      new_seqs[i][j] = entry
  return new_seqs, lengths

def unpad_sequence(seq: List[int], padding_idx: int = 0) -> List[int]:
  '''Remove a padding index.

  >>> unpad_sequence([3, 2, 1, 0, 0, 0])
  [3, 2, 1]
  >>> unpad_sequence([5, 0, 1, 2, 0, 0, 0])
  [5, 1, 2]
  '''
  return [entry for entry in seq if entry != 0]

def ngrams_to_tensor(ngrams: List[Ngram], token2index: Dict[str, int]) \
      -> Tuple[LongTensor, LongTensor]:
  '''Return tensors xss and yss of contexts and targets, respectively.

  :type ngrams: List[Ngram] = List[Tuple[List[str], str]]
  :type token2index: Tuple[LongTensor, LongTensor]
  :return: Tuple (xss, yss) where xss and yss are 2-dimensional tensors
      of size len(ngrams) x len(ngrams[0][0]) resp. len(ngrams) x 1.
  :rtype: LongTensor, LongTensor
  '''
  contexts = [[token2index[token] for token in ngram[0]] for ngram in ngrams]
  targets = [token2index[ngram[1]] for ngram in ngrams]
  return torch.tensor(contexts), torch.tensor(targets) # type: ignore

def tsne_plot(words, vects, num_components = 3):
  ''' Plot a TSNE model. '''
  import matplotlib.pyplot as plt # type: ignore
  from sklearn.manifold import TSNE # type: ignore

  tsne_model = TSNE(
      perplexity=40,
      n_components=num_components,
      init='pca',
      n_iter=2500,
      random_state=23
  )
  new_values = tsne_model.fit_transform(vects)

  x = []; y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])

  plt.figure(figsize=(30, 30))
  for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(words[i],
         xy=(x[i], y[i]),
         xytext=(5, 2),
         textcoords='offset points',
         ha='right',
         va='bottom'
    )
  plt.show()

def pca_plot(tokens, vects):
  '''Find and diplay the 2-component PCA reduction.'''
  from sklearn.decomposition import PCA # type: ignore
  from matplotlib import pyplot # type: ignore

  tokens = [' '+token for token in tokens]
  pca = PCA(n_components = 2)
  result = pca.fit_transform(vects.T)
  pyplot.scatter(result[:,0], result[:,1], s=3, c='firebrick')
  for i, word in enumerate(tokens):
    pyplot.annotate(word, xy=(result[i,0],result[i,1]))
  pyplot.show()

if __name__ == '__main__':
  import inspect
  import doctest

  # find the user defined functions
  _local_functions = [(name,ob) for (name, ob) in sorted(locals().items())\
       if callable(ob) and ob.__module__ == __name__]

  #remove markdown
  #  from the docstring for this module
  globals()['__doc__'] = du.utils._markup(globals()['__doc__'],strip = True)
  #  from the functions (methods are fns in Python3) defined in this module
  for _, _ob in _local_functions:
    if inspect.isfunction(_ob):
      _ob.__doc__ = du.utils._markup(_ob.__doc__,strip = True)
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
        _ob.__dict__[name].__doc__ = du.utils._markup(_meth.__doc__,strip =True)

  # run doctests
  failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)

  # print signatures
  if failures == 0:
    from inspect import signature
    for name, ob in _local_functions:
      print(name,'\n  ', inspect.signature(ob))
