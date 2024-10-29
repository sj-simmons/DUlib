'''display/debug documentation highlighting.

If, in the table below, the word "underline" is underlined, the
word "bold" is bolded, "red" is red, and so on, then highlight-
ing is working. At your command-line, issue the following for a
brief introduction to DUlib and its usage:

  pd -c du

But if, below, the words "underline", "bold", "red", etc. are
hard to read because they are wrapped in escape sequences, then
skip down to the technical note below the bar.

Surrounding a phrase, all on one line, with the specified char-
acter leads to the listed effect.

            exclamation point             !underline!
            a single backquote            `bold`
            a dollar sign                 $red$
            bar                           |blue|
            tilde                         ~cyan~

Also, the following may be highlighted.

            three greater-thans           >>>
            three periods                 ...

                    _____________________


A technical note on breaking out of so-called charset encoding
hell.

If you can easily and normally read the trailing phrase in the
last sentence, then you are not even in charset encoding hell.
In this case, you can simply go ahead and start deep learning.

However, if that line (and, in fact, other words or phrases in
the discussion above) is enclosed in boxes or other weird char-
acters, then you are in so-called charset encoding hell. There
are a number of ways around such rendering issues.

There is a good chance that the problems you are experiencing
are due to your being in IDLE (which is the IDE that ships with
Python) or some other IDE that doesn't play nice with ansi es-
cape sequences.

Recommendation: consume the documentation for DUlib by using
the pd command at your command-line. That way you can enjoy a
few enhancements like bolding and colorizing of certain words.
This aids in quickly and easily finding APIs for this library.

Instead of reading this in an IDE, try issuing the following at
your command-line:

  pd -c du.highlight

Another basic problem might stem from running the command above
from PowerShell in Windows. As of 2022, straight PowerShell
does not have a high quality terminal. Instead, download Wind-
ows Terminal from the MicroSoft App Store.

If you are already using the command-line in some flavor of
Linux or on a Mac and yet still experiencing char escape hell,
then try manually setting the PAGER and/or the TERM environ-
ment variables.

A great value for PAGER is the value: less -r. You can set the
PAGER environment variable on Linux-like systems (and this in-
cludes macOS and WSL) with the bash command:

  export PAGER='less -r'

which you can add to the end of your .bashrc (which lives in
your home directory) if you wish. But be sure to either reboot
or exit and restart your shell session after editing .bashrc;
or to run the command

  source ~/.bashrc

Viable values for TERM are, for example, any one of:

          screen-256color, xterm-256color, or ansi.

You can set TERM in your .bashrc with, e.g.,

  export TERM=xterm-256color

On a newer Mac, the PAGER variable is likely not set. You can
check this with

  echo $PAGER

Set PAGER as above and you should be good to go on a Mac.

Lastly, you can always strip away escape characters when read-
ing this or any page in the documentation for this module by
simply using pd without the -c switch. For example:

  pd du.lib

'''
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
#__doc__ = du.utils._markup(__doc__)
