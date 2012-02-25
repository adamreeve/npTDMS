"""Test reading of example TDMS files"""

import unittest
import sys
import logging
import binascii
import struct
import tempfile
from nptdms import tdms


try:
    long
except NameError:
    # Python 3
    long = int


def string_hexlify(input_string):
    """Return hex string representation of string"""
    return binascii.hexlify(input_string.encode('utf-8')).decode('utf-8')


class TestFile(object):
    """Generate a TDMS file for testing"""

    def __init__(self):
        self.file = tempfile.TemporaryFile()
        self.data = bytes()

    def add_segment(self, metadata, data, toc=None):
        metadata = self.to_bytes(metadata)
        data = self.to_bytes(data)
        if toc is not None:
            lead_in = b'TDSm'
            toc_mask = long(0)
            if "kTocMetaData" in toc:
                toc_mask = toc_mask | long(1) << 1
            if "kTocRawData" in toc:
                toc_mask = toc_mask | long(1) << 3
            if "kTocDAQmxRawData" in toc:
                toc_mask = toc_mask | long(1) << 7
            if "kTocInterleavedData" in toc:
                toc_mask = toc_mask | long(1) << 5
            if "kTocBigEndian" in toc:
                toc_mask = toc_mask | long(1) << 6
            if "kTocNewObjList" in toc:
                toc_mask = toc_mask | long(1) << 2
            lead_in += struct.pack('<i', toc_mask)
            lead_in += self.to_bytes("69 12 00 00")
            next_segment_offset = len(metadata) + len(data)
            raw_data_offset = len(metadata)
            lead_in += struct.pack('<QQ', next_segment_offset, raw_data_offset)
        else:
            lead_in = b''
        self.data += lead_in + metadata + data

    def to_bytes(self, hex_data):
        return binascii.unhexlify(hex_data.replace(" ", "").
                replace("\n", "").encode('utf-8'))

    def load(self):
        self.file.write(self.data)
        self.file.seek(0)
        return tdms.TdmsFile(self.file)


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

    def basic_segment(self):
        """Basic TDMS segment with one group and two channels"""

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
            # Number of raw datata values
            "02 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        data = (
            # Data for segment
            "01 00 00 00"
            "02 00 00 00"
            "03 00 00 00"
            "04 00 00 00"
        )
        return (metadata, data, toc)

    def test_data_read(self):
        """Test reading data"""

        test_file = TestFile()
        test_file.add_segment(*self.basic_segment())
        tdms = test_file.load()

        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 2)
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 3)
        self.assertEqual(data[1], 4)

    def test_property_read(self):
        """Test reading an object property"""

        test_file = TestFile()
        test_file.add_segment(*self.basic_segment())
        tdms = test_file.load()

        object = tdms.object("Group")
        self.assertEqual(object.property("num"), 10)

    def test_no_metadata_segment(self):
        """Add a segment with two channels, then a second
        segment with the same metadata as before,
        so there is only the lead in and binary data"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        test_file.add_segment(metadata, data, toc)
        data = ("05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
        )
        toc = ("kTocRawData")
        test_file.add_segment('', data, toc)
        tdms = test_file.load()

        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))

    def test_new_channel(self):
        """Add a new voltage channel, with the other two channels
        remaining unchanged, so only the new channel is in metadata section"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        test_file.add_segment(metadata, data, toc)
        toc = ("kTocMetaData", "kTocRawData")
        metadata = (
            # Number of objects
            "01 00 00 00"
            # Length of the third object path
            "12 00 00 00")
        metadata += string_hexlify("/'Group'/'Voltage'")
        metadata += (
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
            "00 00 00 00")
        data = (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
            "09 00 00 00"
            "0A 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()

        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))
        data = tdms.channel_data("Group", "Voltage")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [9, 10]))

    def test_larger_channel(self):
        """In the second segment, increase the channel size
        of one channel"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        test_file.add_segment(metadata, data, toc)
        toc = ("kTocMetaData", "kTocRawData")
        metadata = (
            # Number of objects
            "01 00 00 00"
            # Length of the object path
            "13 00 00 00")
        metadata += string_hexlify("/'Group'/'Channel2'")
        metadata += (
            # Length of index information
            "14 00 00 00"
            # Raw data data type
            "03 00 00 00"
            # Dimension
            "01 00 00 00"
            # Number of raw datata values
            "04 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        data = (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
            "09 00 00 00"
            "0A 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()
        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 6)
        self.assertTrue(all(data == [3, 4, 7, 8, 9, 10]))

    def test_remove_channel(self):
        """In the second segment, remove a channel.
        We need to write a new object list in this case"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        test_file.add_segment(metadata, data, toc)
        # Keep toc as it was before, with new object list set
        metadata = (
            # Number of objects
            "01 00 00 00"
            # Length of the object path
            "13 00 00 00")
        metadata += string_hexlify("/'Group'/'Channel1'")
        metadata += (
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
            "00 00 00 00")
        data = (
            "05 00 00 00"
            "06 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()
        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [3, 4]))

    def test_no_lead_in(self):
        """Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        data = data + (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()

        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))

    def test_interleaved(self):
        """Test reading interleaved data"""

        test_file = TestFile()
        (metadata, data, toc) = self.basic_segment()
        toc = toc + ("kTocInterleavedData", )
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()

        data = tdms.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 3)
        data = tdms.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 2)
        self.assertEqual(data[1], 4)

    def test_timestamp_data(self):
        """Test reading contiguous and interleaved timestamp data,
        which isn't read by numpy"""

        metadata = (
            # Number of objects
            "02 00 00 00"
            # Length of the object path
            "17 00 00 00")
        metadata += string_hexlify("/'Group'/'TimeChannel1'")
        metadata += (
            # Length of index information
            "14 00 00 00"
            # Raw data data type
            "44 00 00 00"
            # Dimension
            "01 00 00 00"
            # Number of raw datata values
            "02 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        metadata += (
            "17 00 00 00")
        metadata += string_hexlify("/'Group'/'TimeChannel2'")
        metadata += (
            # Length of index information
            "14 00 00 00"
            # Raw data data type
            "44 00 00 00"
            # Dimension
            "01 00 00 00"
            # Number of raw datata values
            "02 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        data = (
            "00 00 00 00 00 00 00 60"
            "01 00 00 00 00 00 00 00"
            "00 00 00 00 00 00 00 60"
            "02 00 00 00 00 00 00 00"
            "00 00 00 00 00 00 00 60"
            "03 00 00 00 00 00 00 00"
            "00 00 00 00 00 00 00 60"
            "04 00 00 00 00 00 00 00"
        )

        test_file = TestFile()
        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()
        channel_data = tdms.channel_data("Group", "TimeChannel1")
        self.assertEqual(len(channel_data), 2)
        self.assertEqual(channel_data[0][0], 1)
        self.assertEqual(channel_data[1][0], 2)
        # Read fraction of second
        self.assertEqual(channel_data[0][1], 0.375)
        channel_data = tdms.channel_data("Group", "TimeChannel2")
        self.assertEqual(len(channel_data), 2)
        self.assertEqual(channel_data[0][0], 3)
        self.assertEqual(channel_data[1][0], 4)

        # Now test it interleaved
        toc = toc + ("kTocInterleavedData", )
        test_file = TestFile()
        test_file.add_segment(metadata, data, toc)
        tdms = test_file.load()
        channel_data = tdms.channel_data("Group", "TimeChannel1")
        self.assertEqual(len(channel_data), 2)
        self.assertEqual(channel_data[0][0], 1)
        self.assertEqual(channel_data[1][0], 3)
        channel_data = tdms.channel_data("Group", "TimeChannel2")
        self.assertEqual(len(channel_data), 2)
        self.assertEqual(channel_data[0][0], 2)
        self.assertEqual(channel_data[1][0], 4)


if __name__ == '__main__':
    unittest.main()
