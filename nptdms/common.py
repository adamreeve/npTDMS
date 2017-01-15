from collections import namedtuple
import numpy as np


try:
    long
except NameError:
    # Python 3
    long = int


toc_properties = {
    'kTocMetaData': (long(1) << 1),
    'kTocRawData': (long(1) << 3),
    'kTocDAQmxRawData': (long(1) << 7),
    'kTocInterleavedData': (long(1) << 5),
    'kTocBigEndian': (long(1) << 6),
    'kTocNewObjList': (long(1) << 2)
}
