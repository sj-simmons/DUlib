#!/usr/bin/env python3
'''NLP utilities and recurrent net classes, per the DL@DU project.

Notes:
  - You can run this file at the commandline to quickly obtain the
    signatures of the local functions defined herein:
        python3 wordlib.py
  - Also, you can type check these functions with:
        mypy worldlib.py
Todo:
  - Consider calling function like e.g. line2chars something like line_to_chars
    and then keep using token2index etc for a dict in actual programs.
  - also change make_tokens to line_to_tokens  and make_token to tokenize, so
    to keep with make_blah as returning a dict.
'''
import glob
import io
import os
from string import punctuation
from typing import List, Dict, Tuple, Sequence, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'Simmons'
__version__ = '0.3dev'
__status__ = 'Development'
__date__ = '___'

# define some type aliases
Ngram = Tuple[List[str], str]
Tensor = torch.Tensor
LongTensor = torch.LongTensor

def file2lines(file_name: str) -> List[str]:
  '''Read in lines of a file to a list.

  Args:
    file_name (list): The name of the file to read.

  Returns:
    List[str]. A list whose items are the lines of the file.
  '''
  with open(file_name) as f:
    corpus = f.read()
  return [line for line in corpus.split('\n') if line != '']

def dir2lines(dir_name: str, small: int = -1) -> Dict[str, List]:
  '''For each file (with extension '.txt') in a directory, read
  its lines into a list.

  Args:
    small (int): Read only this many lines from each file.

  Returns:
    Dict[str, List[str]]. A dict of the lists indexed by the
        filename bases.
  '''
  d = {}
  for filename in glob.glob(dir_name+'/*.txt'):
    basename = os.path.splitext(os.path.basename(filename))[0]
    d[basename.lower()] = file2lines(filename)[:small]
  return d

def invertListDict(d: Dict[Any, List[Any]]) -> Dict[Any, Any]:
  '''Given a dict with values that are lists, return a dict whose
  keys are the elements of those lists and values the apropriate key.

  >>> d = {'one': [1.2, 1.34], 'two': [2.0, 2.5]}
  >>> d_inverted = {1.2: 'one', 1.34: 'one', 2.0: 'two', 2.5: 'two'}
  >>> invertListDict(d) == d_inverted
  True
  '''
  return {item: k for k, lst in d.items() for item in lst}

def make_token(word: str) -> str:
  '''Return the token for a word.

  >>> make_token("You're,")
  "you're"
  '''
  return word.strip(punctuation).lower()

def make_tokens(line: str) -> List[str]:
  '''Split line on whitespace and return the resulting list of tokens.

  :rtype: List[str]

  >>> make_tokens('San ti wenti; Lang tu tong')
  ['san', 'ti', 'wenti', 'lang', 'tu', 'tong']
  >>> make_tokens('??')
  []
  '''
  tokens = [make_token(s) for s in line.split()]
  return [token for token in tokens if token != '']

def line2tokens(lst: List[str]) -> List[str]:
  return [make_tokens(item) for item in lst]

def make_token2index(tokens: Union[List[str],List[List[str]]],pad: str = None)\
    -> Dict[str, int]:
  '''Return a dict whose keys are the unique tokens and whose values
  are from the set {0,1,...,len(dict)}.

  Args:
    tokens (Union[List[str], List[List[str]]]): List or list of lists of
        tokens the elements of which are the tokens of the corpus.
    pad (str): Add this string to the corpus with index 0.

  Returns:
    Dict[str, int]. A dict (often called vocab) mapping tokens to indices.

  >>> make_token2index([['a', 'z', 'i'], ['r', 'z'], ['i']]) # doctest: +SKIP
  '''
  if isinstance(tokens[0], list):
    tokens = [item for sublist in tokens for item in sublist]
  vocab = set(tokens)
  if pad:
    return {token: i for i, token in enumerate([pad] + sorted(vocab))}
  else:
    return {token: i for i, token in enumerate(sorted(vocab))}


## Deprecate this in favor of above
#def make_token2index(lines: List[str], pad: str = None) -> Dict[str, int]:
#  '''Return a dict whose keys are the unique tokens and whose values
#  are from the set {0,1,...,len(dict)}.
#
#  Args:
#    lines (List[str]): List whose elements are the lines of the corpus.
#    pad (str): Add this string to the corpus with index 0.
#
#  Returns:
#    Dict[str, int]. A dict (often called vocab) mapping tokens to indices.
#
#  >>> vocab = make_token2index(['A screaming comes across','the sky.'])
#  >>> vocab == {'a':0,'across':1,'comes':2,'screaming':3,'sky':4,'the':5}
#  True
#  >>> vocab=make_token2index(['A screaming comes across','the sky.'], '<PAD>')
#  >>> out={'<PAD>':0,'a':1,'across':2,'comes':3,'screaming':4,'sky':5,'the':6}
#  >>> vocab == out
#  True
#  '''
#  print("make_token2index will be DEPRECATED, use make_token2index_ for now")
#  vocab = set()
#  for line in lines:
#    vocab.update(make_tokens(line))
#  vocab.discard('')
#  if pad:
#    return {token: i for i, token in enumerate([pad] + sorted(vocab))}
#  else:
#    return {token: i for i, token in enumerate(sorted(vocab))}

def make_ngram(tokens: List[str]) -> Ngram:
  '''Return the ngram for a list of tokens.

  :type tokens: List[str]
  :rtype: Ngram = Tuple[List[str], str]

  >>> make_ngram(['one', 'giant', 'leap', 'for', 'mankind'])
  (['one', 'giant', 'leap', 'for'], 'mankind')
  '''
  return tokens[:-1], tokens[-1]

def line2chars(line: str) -> List[str]:
  '''Return list of characters in a line.

  Args:
    line (str): The line to be processed.

  Returns:
    List[str]: The list of (lowercased) characters.

  >>> line2chars('Coltrane')
  ['c', 'o', 'l', 't', 'r', 'a', 'n', 'e']
  >>> line2chars("L'Hopital")
  ['l', "'", 'h', 'o', 'p', 'i', 't', 'a', 'l']
  '''
  return list(line.lower())

def line2ngrams(line: str, n: int) -> List[Ngram]:
  '''Return a list of ngrams for line.

  :rtype: List[Ngram] = List[Tuple[List[str], str]]

  >>> line2ngrams('Yi er san si.', 3)
  [(['yi', 'er'], 'san'), (['er', 'san'], 'si')]
  >>> line2ngrams('Yi er san si.', 4)
  [(['yi', 'er', 'san'], 'si')]
  >>> line2ngrams('??', 4)
  []
  '''
  tokens = make_tokens(line)
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
  line_wise_ngrams = [line2ngrams(line, n) for line in lines]
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

def ngrams2tensor(ngrams: List[Ngram], token2index: Dict[str, int]) \
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

class SimpleRNN(nn.Module):
  def __init__(self, n_in, enc_dim, n_hid, n_out,padding_idx,device:str ='cpu'):
    super(SimpleRNN, self).__init__()
    self.n_in = n_in
    self.n_hid = n_hid
    self.device = device
    self.padding_idx = padding_idx
    self.hidden = torch.zeros(1, n_hid).to(device)
    self.comb2hid = nn.Linear(n_in + n_hid, n_hid)
    self.comb2out = nn.Linear(n_in + n_hid, n_out)

  def forward(self, xss, lengths = None):
    xs = unpad_sequence(xss.squeeze(0), self.padding_idx)
    for x_ in xs:
      x_one_hot = F.one_hot(x_, self.n_in).float().unsqueeze(0)
      combined = torch.cat((x_one_hot, self.hidden),dim = 1)
      self.hidden = self.comb2hid(combined)
    logit = self.comb2out(combined)
    self.hidden = torch.zeros(1, self.n_hid).to(self.device)
    return torch.log_softmax(logit,dim=1)

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
