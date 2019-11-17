from setuptools import setup

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
  name='DUlib',
  url='https://github.com/sj-simmons/DUlib',
  download_url='https://github.com/sj-simmons/DUlib/archive/v0.3.tar.gz',
  author='SSimmons',
  author_email='ssimmons@drury.edu',
  packages=['du'],
  #install_requires=['torch>=1.2.0+cpu', 'scipy', 'matplotlib<3.1', 'scikit-image<0.16'],
  #torch has been removed from the requirements so we assume the user had
  #already installed their desired version of torch.
  install_requires=['scipy', 'matplotlib<3.1', 'scikit-image<0.16'],
  version='0.4',
  license='Apache 2.0',
  description='tools from the DL@DU Project',
  long_description=long_description,
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Education',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Topic :: Software Development',
      'License :: OSI Approved :: Apache Software License'
  ]
)
