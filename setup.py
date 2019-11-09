from setuptools import setup

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
  name='DUlib',
  url='https://github.com/sj-simmons/DUlib',
  author='SSimmons',
  author_email='ssimmons@drury.edu',
  packages=['du'],
  #install_requires=['torch'],
  version='0.3dev',
  license='Apache 2.0',
  description='DL tools courtesy of the DL@DU project',
  long_description=long_description
)
