"""Module for reading binary TDMS files produced by LabView"""

from __future__ import absolute_import


# Make version number available
from .version import __version_info__, __version__

# Export public objects
from .tdms import TdmsFile, TdmsObject
