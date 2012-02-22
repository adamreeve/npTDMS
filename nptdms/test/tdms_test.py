"""Test reading in an example TDMS file"""

import unittest
from nptdms import tdms


class TestFile(object):
    """Provide a file like interface to example TDMS file"""

    def __init__(self, interleaved=False):
        # tdms header
        tdms_contents = "54 44 53 6D"

        # toc mask
        if interleaved:
            tdms_contents += "2E 00 00 00"
        else:
            tdms_contents += "0E 00 00 00"

        tdms_contents += (
            # version number
            "69 12 00 00"
            # next segment offset
            "73 00 00 00 00 00 00 00"
            # raw data offset
            "6B 00 00 00 00 00 00 00"
            # Number of objects
            "02 00 00 00"
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
            #Data type of property
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
            # Number of properties (0)
            "00 00 00 00"
            # Data for segment
            "01 00 00 00"
            "02 00 00 00"
        )
        self.data = tdms_contents.replace(" ", "").decode('hex')
        self.pos = 0

    def read(self, length):
        data = self.data[self.pos:self.pos + length]
        self.pos += length
        return data

    def tell(self):
        return self.pos

    def seek(self, pos):
        self.pos = pos


class TdmsTestFile(tdms.TdmsFile):
    """Reimplement the init routine so we can read from our example file"""

    def __init__(self, tdms_file):
        self.segments = []
        self.objects = {}
        self._read_segments(tdms_file)


class TDMSTestClass(unittest.TestCase):
    def test_basic_read(self):
        tdms_file = TestFile()
        t = TdmsTestFile(tdms_file)
        data = t.channel_data("Group", "Channel1")
        assert(len(data) == 2)
        assert(data[0] == 1)
        assert(data[1] == 2)


if __name__ == '__main__':
    unittest.main()
