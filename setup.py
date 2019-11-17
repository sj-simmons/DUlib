from setuptools import setup, find_packages

def readme():
  with open('README.rst', 'r') as fh:
    return fh.read()

setup(
  name='DUlib',
  url='https://github.com/sj-simmons/DUlib',
  download_url='https://github.com/sj-simmons/DUlib/archive/v0.4.tar.gz',
  author='SSimmons',
  author_email='ssimmons@drury.edu',
  packages=find_packages(),
  #install_requires=['torch>=1.2.0+cpu', 'scipy', 'matplotlib<3.1', 'scikit-image<0.16'],
  #We do not include torch in install_requires since then users can install
  #the appropriate verion of torch (e.gl, cpu vs. gpu) for their machine.
  install_requires=['scipy', 'matplotlib<3.1', 'scikit-image<0.16'],
  version='0.6',
  license='Apache 2.0',
  description='tools from the DL@DU Project',
  long_description=readme(),
  include_package_data=True,
  zip_safe=False,
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
