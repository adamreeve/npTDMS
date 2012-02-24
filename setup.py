from distutils.core import setup

setup(
  name = 'npTDMS',
  version = '0.1.1',
  description = ("Cross-platform, NumPy based module for reading "
    "TDMS files produced by LabView."),
  author = 'Adam Reeve',
  author_email = 'adreeve@gmail.com',
  url = 'http://pypi.python.org/pypi/npTDMS',
  packages = ['nptdms', 'nptdms.test'],
  long_description=open('README.txt').read(),
  license = 'LGPL',
  install_requires = ['numpy'])
