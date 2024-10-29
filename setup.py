from setuptools import setup, find_packages

def readme():
  with open('README.rst', 'r') as fh:
    return fh.read()

setup(
  entry_points={
      'console_scripts': [
          'pd = du._pydoc:cli',
          'dulib_linreg = du.examples:simple_linear_regression',
          'dulib_linreg_anim = du.examples:simple_linear_regression_animate',
          'dulib_polyreg = du.examples:simple_polynomial_regression',
          'dulib_polyreg_anim = du.examples:simple_polynomial_regression_animate',
      ],
      'gui_scripts': [],
  },
  name='DUlib',
  url='https://github.com/sj-simmons/DUlib',
  download_url='https://github.com/sj-simmons/DUlib/archive/v0.9.96.tar.gz',
  author='Scott Simmons',
  author_email='ssimmons@drury.edu',
  packages=find_packages(),
  #install_requires=['torch>=1.2.0+cpu', 'scipy', 'matplotlib<3.1', 'scikit-image<0.16'],
  #We do not include torch in install_requires since then users can install
  #the appropriate verion of torch (e.gl, cpu vs. gpu) for their machine.
  #Do the same for matplotlib
  #install_requires=['matplotlib<3.1'],
  python_requires='>=3.6',
  install_requires=[],
  version='0.9.96',
  license='Apache 2.0',
  description='courtesy of the DL@DU Project',
  long_description=readme(),
  include_package_data=True,
  zip_safe=False,
  project_urls={'Upstream Repository': 'https://github.com/sj-simmons/DUlib'},
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Intended Audience :: Education',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Topic :: Software Development',
      'License :: OSI Approved :: Apache Software License'
  ]
)
