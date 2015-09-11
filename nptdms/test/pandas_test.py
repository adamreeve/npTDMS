"""Test exporting TDMS data to Pandas"""

import unittest
import sys
import logging
from datetime import datetime
try:
    import pytz
except ImportError:
    pytz = None

from nptdms import tdms
from .tdms_test import (
    TestFile,
    basic_segment,
    string_hexlify,
    hexlify_value
)


if pytz:
    timezone = pytz.utc
else:
    timezone = None


def within_tol(a, b, tol=1.0e-10):
    return abs(a - b) < tol


def timed_segment():
    """TDMS segment with one group and two channels,
    each with time properties"""

    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "03 00 00 00"
        # Length of the first object path
        "08 00 00 00"
        # Object path (/'Group')
        "2F 27 47 72"
        "6F 75 70 27"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "02 00 00 00"
        # Name length
        "04 00 00 00"
        # Property name (prop)
        "70 72 6F 70"
        # Property data type (string)
        "20 00 00 00"
        # Length of string value
        "05 00 00 00"
        # Value
        "76 61 6C 75 65"
        # Length of second property name
        "03 00 00 00"
        # Property name (num)
        "6E 75 6D"
        # Data type of property
        "03 00 00 00"
        # Value
        "0A 00 00 00"
        # Length of the second object path
        "13 00 00 00"
        # Second object path (/'Group'/'Channel1')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 31 27"
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties
        "03 00 00 00"
        # Set time properties for the first channel
        "0F 00 00 00" +
        string_hexlify('wf_start_offset') +
        "0A 00 00 00" +
        hexlify_value("<d", 2.0) +
        "0C 00 00 00" +
        string_hexlify('wf_increment') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.1) +
        "0D 00 00 00" +
        string_hexlify('wf_start_time') +
        "44 00 00 00" +
        hexlify_value("<Q", 0) +
        hexlify_value("<q", 3524551547) +
        # Length of the third object path
        "13 00 00 00"
        # Third object path (/'Group'/'Channel2')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 32 27"
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties
        "03 00 00 00"
        # Set time properties for the second channel
        "0F 00 00 00" +
        string_hexlify('wf_start_offset') +
        "0A 00 00 00" +
        hexlify_value("<d", 2.0) +
        "0C 00 00 00" +
        string_hexlify('wf_increment') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.1) +
        "0D 00 00 00" +
        string_hexlify('wf_start_time') +
        "44 00 00 00" +
        hexlify_value("<Q", 0) +
        hexlify_value("<q", 3524551547))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "03 00 00 00"
        "04 00 00 00"
    )
    return (metadata, data, toc)


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

    def test_file_as_dataframe(self):
        """Test converting file to Pandas dataframe"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.as_dataframe()

        self.assertEqual(len(df), 2)
        self.assertIn("/'Group'/'Channel1'", df.keys())
        self.assertIn("/'Group'/'Channel2'", df.keys())

        self.assertTrue((df["/'Group'/'Channel1'"] == [1, 2]).all())

    def test_file_as_dataframe_without_time(self):
        """Converting file to dataframe with time index should raise when
        time properties aren't present"""

        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        with self.assertRaises(KeyError):
            df = tdmsData.as_dataframe(time_index=True)

    def test_file_as_dataframe_with_time(self):
        """Test converting file to Pandas dataframe with a time index"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.as_dataframe(time_index=True)

        self.assertEqual(len(df.index), 2)
        self.assertTrue(within_tol(df.index[0], 2.0))
        self.assertTrue(within_tol(df.index[1], 2.1))

    def test_file_as_dataframe_with_absolute_time(self):
        """Convert file to Pandas dataframe with absolute time index"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.as_dataframe(time_index=True, absolute_time=True)

        expected_start = datetime(2015, 9, 8, 10, 5, 49, tzinfo=timezone)
        self.assertTrue((df.index == expected_start)[0])

    def test_channel_as_dataframe(self):
        """Convert a channel to dataframe"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.object("Group", "Channel2").as_dataframe()
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.keys()), 1)
        self.assertIn("/'Group'/'Channel2'", df.keys())
        self.assertTrue((df["/'Group'/'Channel2'"] == [3, 4]).all())

    def test_channel_as_dataframe_with_time(self):
        """Convert a channel to dataframe with a time index"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.object("Group", "Channel2").as_dataframe()
        self.assertEqual(len(df.index), 2)
        self.assertTrue(within_tol(df.index[0], 2.0))
        self.assertTrue(within_tol(df.index[1], 2.1))

    def test_channel_as_dataframe_without_time(self):
        """Converting channel to dataframe should raise when
        time properties aren't present"""

        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        with self.assertRaises(KeyError):
            df = tdmsData.object("Group", "Channel1").as_dataframe()

    def test_channel_as_dataframe_with_absolute_time(self):
        """Convert channel to Pandas dataframe with absolute time index"""

        test_file = TestFile()
        test_file.add_segment(*timed_segment())
        tdmsData = test_file.load()

        df = tdmsData.object("Group", "Channel1").as_dataframe(
            absolute_time=True)

        expected_start = datetime(2015, 9, 8, 10, 5, 49, tzinfo=timezone)
        self.assertTrue((df.index == expected_start)[0])


if __name__ == '__main__':
    unittest.main()
