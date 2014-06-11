import os

from setuptools import setup


def read_version():
    here = os.path.abspath(os.path.dirname(__file__))
    version_path = os.path.sep.join((here, "nptdms", "version.py"))
    v_globals = {}
    v_locals = {}
    exec(open(version_path).read(), v_globals, v_locals)
    return v_locals['__version__']


setup(
  name = 'npTDMS',
  version = read_version(),
  description = ("Cross-platform, NumPy based module for reading "
    "TDMS files produced by LabView."),
  author = 'Adam Reeve',
  author_email = 'adreeve@gmail.com',
  url = 'https://github.com/adamreeve/npTDMS',
  packages = ['nptdms', 'nptdms.test'],
  long_description=open('README.rst').read(),
  license = 'LGPL',
  classifiers = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
  ],
  install_requires = ['numpy'],
  entry_points = """
  [console_scripts]
  tdmsinfo=nptdms.tdmsinfo:main
  """
)
