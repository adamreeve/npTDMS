"""Test reading of TDMS files with DAQmx data
"""

import logging
import unittest
import numpy as np

from nptdms import tdms
from nptdms.test.util import TestFile


class DaqmxDataTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

    def test_single_channel_i16(self):
        """ Test loading a DAQmx file with a single channel of I16 data
        """

        test_file = TestFile()
        test_file.add_segment(*daqmx_segment())
        tdms_data = test_file.load()

        data = tdms_data.object("Group", "Channel1").raw_data

        self.assertEqual(data.dtype, np.int16)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 2)
        self.assertEqual(data[2], 3)
        self.assertEqual(data[3], 4)


def daqmx_segment():
    """TDMS segment with DAQmx data"""

    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocDAQmxRawData")
    metadata = (
        # Number of objects
        "03 00 00 00"
        # Length of the first object path
        "01 00 00 00"
        # Object path (/)
        "2F"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00"
        # Length of the seond object path
        "08 00 00 00"
        # Object path (/'Group')
        "2F 27 47 72"
        "6F 75 70 27"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00"
        # Length of the third object path
        "13 00 00 00"
        # Second object path (/'Group'/'Channel1')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 31 27"
        # Raw data index (DAQmx)
        "69 12 00 00"
        # Data type (DAQmx)
        "FF FF FF FF"
        # Array  dimension
        "01 00 00 00"
        # Number of values (chunk size)
        "04 00 00 00"
        "00 00 00 00"
        # Number of scalers
        "01 00 00 00"
        # DAQmx data type (I16, type ids don't match TDMS types)
        "03 00 00 00"
        # Raw buffer index
        "00 00 00 00"
        # Raw byte offset
        "00 00 00 00"
        # Sample format bitmap (don't know what this is for...)
        "00 00 00 00"
        # Scale ID
        "00 00 00 00"
        # Raw data width vector size
        "01 00 00 00"
        # First raw data width in bytes (2)
        "02 00 00 00"
        # Number of properties (0)
        "00 00 00 00"
    )
    data = (
        # Data for segment
        "01 00"
        "02 00"
        "03 00"
        "04 00"
    )
    return (metadata, data, toc)
