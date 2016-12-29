"""Test reading of example TDMS files"""

import unittest
import sys
import logging
import binascii
import struct
import tempfile
from datetime import datetime
import os
import numpy as np

from nptdms import tdms

_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'

try:
    long
except NameError:
    # Python 3
    long = int


def string_hexlify(input_string):
    """Return hex string representation of string"""
    return binascii.hexlify(input_string.encode('utf-8')).decode('utf-8')


def hexlify_value(struct_type, value):
    """Return hex string representation of a value"""
    return binascii.hexlify(struct.pack(struct_type, value)).decode('utf-8')


def basic_segment():
    """Basic TDMS segment with one group and two channels"""

    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "04 00 00 00"
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
        # Number of data values
        "02 00 00 00"
        "00 00 00 00"
        # Set time properties for the second channel
        "02 00 00 00"
        "0F 00 00 00" +
        string_hexlify('wf_start_offset') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.0) +
        "0C 00 00 00" +
        string_hexlify('wf_increment') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.1) +
        # Length of the object path
        "01 00 00 00"
        # Object path (/)
        "2F"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00"
    )
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "03 00 00 00"
        "04 00 00 00"
    )
    return (metadata, data, toc)


class TestFile(object):
    """Generate a TDMS file for testing"""

    def __init__(self):
        self._tempfile = tempfile.NamedTemporaryFile()
        self.file = self._tempfile.file
        self.data = bytes()

    def add_segment(self, metadata, data, toc=None, incomplete=False):
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
            if incomplete:
                lead_in += self.to_bytes('FF' * 8)
            else:
                lead_in += struct.pack('<Q', next_segment_offset)
            lead_in += struct.pack('<Q', raw_data_offset)
        else:
            lead_in = b''
        self.data += lead_in + metadata + data

    def to_bytes(self, hex_data):
        return binascii.unhexlify(
            hex_data.replace(" ", "").replace("\n", "").encode('utf-8'))

    def load(self, *args, **kwargs):
        self.file.write(self.data)
        self.file.seek(0)
        return tdms.TdmsFile(self.file, *args, **kwargs)


class TDMSTestClass(unittest.TestCase):
    def setUp(self):
        logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

    def test_data_read(self):
        """Test reading data"""

        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 2)
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 3)
        self.assertEqual(data[1], 4)

    def test_property_read(self):
        """Test reading an object property"""

        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        object = tdmsData.object("Group")
        self.assertEqual(object.property("num"), 10)

    def test_no_metadata_segment(self):
        """Add a segment with two channels, then a second
        segment with the same metadata as before,
        so there is only the lead in and binary data"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        data = (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
        )
        toc = ("kTocRawData")
        test_file.add_segment('', data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))

    def test_no_metadata_object(self):
        """Re-use an object without setting any new metadata and
        re-using the data structure"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        data = (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
        )
        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
        # Use same object list, but set raw data index to 0
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
            "00 00 00 00"
            # Length of the second object path
            "13 00 00 00"
            # Second object path (/'Group'/'Channel1')
            "2F 27 47 72"
            "6F 75 70 27"
            "2F 27 43 68"
            "61 6E 6E 65"
            "6C 31 27"
            # Raw data index meaning repeat previous data structure
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
            # Raw data index meaning repeat previous data structure
            "00 00 00 00"
            # Number of properties
            "00 00 00 00")
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))

    def test_new_channel(self):
        """Add a new voltage channel, with the other two channels
        remaining unchanged, so only the new channel is in metadata section"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
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
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))
        data = tdmsData.channel_data("Group", "Voltage")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [9, 10]))

    def test_larger_channel(self):
        """In the second segment, increase the channel size
        of one channel"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
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
        tdmsData = test_file.load()
        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 6)
        self.assertTrue(all(data == [3, 4, 7, 8, 9, 10]))

    def test_remove_channel(self):
        """In the second segment, remove a channel.
        We need to write a new object list in this case"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
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
        tdmsData = test_file.load()
        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [3, 4]))

    def test_no_lead_in(self):
        """Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        data = data + (
            "05 00 00 00"
            "06 00 00 00"
            "07 00 00 00"
            "08 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [1, 2, 5, 6]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 4)
        self.assertTrue(all(data == [3, 4, 7, 8]))

    def test_interleaved(self):
        """Test reading interleaved data"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        toc = toc + ("kTocInterleavedData", )
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 3)
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 2)
        self.assertEqual(data[1], 4)

    def test_less_data_than_expected(self):
        """Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks,
        but the extra chunk does not have as much data as expected."""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        data = data + (
            "05 00 00 00"
            "06 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 3)
        self.assertTrue(all(data == [1, 2, 5]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 3)
        self.assertTrue(all(data == [3, 4, 6]))

    def test_less_data_than_expected_interleaved(self):
        """Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks,
        but the extra chunk does not have as much data as expected.
        This also uses interleaved data"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        toc = toc + ("kTocInterleavedData", )
        data = data + (
            "05 00 00 00"
            "06 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 3)
        self.assertTrue(all(data == [1, 3, 5]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 3)
        self.assertTrue(all(data == [2, 4, 6]))

    def test_timestamp_data(self):
        """Test reading contiguous and interleaved timestamp data,
        which isn't read by numpy"""

        times = [
            datetime(2012, 8, 23, 0, 0, 0, 123, tzinfo=tdms.timezone),
            datetime(2012, 8, 23, 1, 2, 3, 456, tzinfo=tdms.timezone),
            datetime(2012, 8, 23, 12, 0, 0, 0, tzinfo=tdms.timezone),
            datetime(2012, 8, 23, 12, 2, 3, 9999, tzinfo=tdms.timezone),
        ]

        def total_seconds(td):
            # timedelta.total_seconds() only added in 2.7
            return td.seconds + td.days * 24 * 3600

        seconds = [
            total_seconds(
                t - datetime(1904, 1, 1, 0, 0, 0, tzinfo=tdms.timezone))
            for t in times]
        fractions = [
            int(float(t.microsecond) * 2 ** 58 / 5 ** 6)
            for t in times]

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
        data = ""
        for f, s in zip(fractions, seconds):
            data += hexlify_value("<Q", f)
            data += hexlify_value("<q", s)

        def assertTimeEqual(a, b):
            """Check times are equal, allowing for small difference
            in microseconds due to floating point math being used
            """

            self.assertEqual(
                a.replace(microsecond=0), b.replace(microsecond=0))
            assert(abs(a.microsecond - b.microsecond) < 10)

        test_file = TestFile()
        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()
        channel_data = tdmsData.channel_data("Group", "TimeChannel1")
        self.assertEqual(len(channel_data), 2)
        assertTimeEqual(channel_data[0], times[0])
        assertTimeEqual(channel_data[1], times[1])
        # Read fraction of second
        channel_data = tdmsData.channel_data("Group", "TimeChannel2")
        self.assertEqual(len(channel_data), 2)
        assertTimeEqual(channel_data[0], times[2])
        assertTimeEqual(channel_data[1], times[3])

        # Now test it interleaved
        toc = toc + ("kTocInterleavedData", )
        test_file = TestFile()
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()
        channel_data = tdmsData.channel_data("Group", "TimeChannel1")
        self.assertEqual(len(channel_data), 2)
        assertTimeEqual(channel_data[0], times[0])
        assertTimeEqual(channel_data[1], times[2])
        channel_data = tdmsData.channel_data("Group", "TimeChannel2")
        self.assertEqual(len(channel_data), 2)
        assertTimeEqual(channel_data[0], times[1])
        assertTimeEqual(channel_data[1], times[3])

    def test_time_track(self):
        """Add a time track to waveform data"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        obj = tdmsData.object("Group", "Channel2")
        time = obj.time_track()
        self.assertEqual(len(time), len(obj.data))
        epsilon = 1.0E-15
        self.assertTrue(abs(time[0]) < epsilon)
        self.assertTrue(abs(time[1] - 0.1) < epsilon)

    def test_no_data_section(self):
        """kTocRawData is set but data length is zero

        Keep first segment the same but add a second
        segment with no data."""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        toc = ("kTocMetaData", "kTocRawData")
        metadata = (
            # Number of objects
            "02 00 00 00"
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
            "00 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        metadata += (
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
            "00 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        data = ""
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()
        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [1, 2]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [3, 4]))

    def test_repeated_object_without_data(self):
        """Repeated objects with no data in new segment

        A new object is also added with new data in order
        to trigger a bug with the chunk size calculation.
        """

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
        metadata = (
            # Number of objects
            "03 00 00 00"
            # Length of the object path
            "13 00 00 00")
        metadata += string_hexlify("/'Group'/'Channel1'")
        metadata += (
            # Raw data index
            "FF FF FF FF"
            # Number of properties (0)
            "00 00 00 00")
        metadata += (
            # Length of the object path
            "13 00 00 00")
        metadata += string_hexlify("/'Group'/'Channel2'")
        metadata += (
            # Raw data index
            "FF FF FF FF"
            # Number of properties (0)
            "00 00 00 00"
            # Length of the third object path
            "13 00 00 00")
        metadata += string_hexlify("/'Group'/'Channel3'")
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
            "01 00 00 00"
            "02 00 00 00"
        )
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()
        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [1, 2]))
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [3, 4]))
        data = tdmsData.channel_data("Group", "Channel3")
        self.assertEqual(len(data), 2)
        self.assertTrue(all(data == [1, 2]))

    def test_memmapped_read(self):
        """Test reading data into memmapped arrays"""

        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load(memmap_dir=tempfile.gettempdir())

        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 2)
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 3)
        self.assertEqual(data[1], 4)

    def test_incomplete_data(self):
        """Test incomplete last segment, eg. if LabView crashed"""

        test_file = TestFile()
        (metadata, data, toc) = basic_segment()
        test_file.add_segment(metadata, data, toc)
        # Add second, incomplete segment
        test_file.add_segment(metadata, data, toc, incomplete=True)
        tdmsData = test_file.load()

        # Eventually we might want to attempt to read the data
        # from the incomplete segment, but for now just make
        # sure we can read the data from previous segments
        data = tdmsData.channel_data("Group", "Channel1")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 2)
        data = tdmsData.channel_data("Group", "Channel2")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], 3)
        self.assertEqual(data[1], 4)

    def test_string_data(self):
        """Test reading a file with string data"""

        strings = ["abcdefg", "qwertyuiop"]

        test_file = TestFile()
        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
        metadata = (
            # Number of objects
            "01 00 00 00"
            # Length of the object path
            "18 00 00 00")
        metadata += string_hexlify("/'Group'/'StringChannel'")
        metadata += (
            # Length of index information
            "1C 00 00 00"
            # Raw data data type
            "20 00 00 00"
            # Dimension
            "01 00 00 00"
            # Number of raw datata values
            "02 00 00 00"
            "00 00 00 00"
            # Number of bytes in data
            "19 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00")
        data = (
            "07 00 00 00"  # index to after first string
            "11 00 00 00"  # index to after second string
        )
        for string in strings:
            data += string_hexlify(string)
        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        data = tdmsData.channel_data("Group", "StringChannel")
        self.assertEqual(len(data), len(strings))
        for expected, read in zip(strings, data):
            self.assertEqual(expected, read)

    def test_slash_and_space_in_name(self):
        """Test name like '01/02/03 something'"""

        group_1_name = "01/02/03 something"
        channel_1_name = "04/05/06 another thing"
        group_2_name = "01/02/03 a"
        channel_2_name = "04/05/06 b"

        test_file = TestFile()

        toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")

        # Number of objects
        metadata = "04 00 00 00"

        for group in [group_1_name, group_2_name]:
            path = "/'{0}'".format(group)
            metadata += hexlify_value('<l', len(path))
            metadata += string_hexlify(path)
            metadata += (
                # Raw data index
                "FF FF FF FF"
                # Number of properties (0)
                "00 00 00 00"
            )
        for (group, channel) in [
                (group_1_name, channel_1_name),
                (group_2_name, channel_2_name)]:
            path = "/'{0}'/'{1}'".format(group, channel)
            metadata += hexlify_value('<l', len(path))
            metadata += string_hexlify(path)
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
                "00 00 00 00"
            )

        data = (
            # Data for segment
            "01 00 00 00"
            "02 00 00 00"
            "03 00 00 00"
            "04 00 00 00"
        )

        test_file.add_segment(metadata, data, toc)
        tdmsData = test_file.load()

        self.assertEqual(len(tdmsData.groups()), 2)
        self.assertEqual(len(tdmsData.group_channels(group_1_name)), 1)
        self.assertEqual(len(tdmsData.group_channels(group_2_name)), 1)
        data_1 = tdmsData.channel_data(group_1_name, channel_1_name)
        self.assertEqual(len(data_1), 2)
        data_2 = tdmsData.channel_data(group_2_name, channel_2_name)
        self.assertEqual(len(data_2), 2)

    def test_root_object_paths(self):
        """Test the group and channel properties for the root object"""
        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        obj = tdmsData.object()
        self.assertEqual(obj.group, None)
        self.assertEqual(obj.channel, None)

    def test_group_object_paths(self):
        """Test the group and channel properties for a group"""
        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        obj = tdmsData.object("Group")
        self.assertEqual(obj.group, "Group")
        self.assertEqual(obj.channel, None)

    def test_channel_object_paths(self):
        """Test the group and channel properties for a group"""
        test_file = TestFile()
        test_file.add_segment(*basic_segment())
        tdmsData = test_file.load()

        obj = tdmsData.object("Group", "Channel1")
        self.assertEqual(obj.group, "Group")
        self.assertEqual(obj.channel, "Channel1")

    def test_labview_file(self):
        """Test reading a file that was created by LabVIEW"""
        tf = tdms.TdmsFile(_data_dir + '/Digital_Input.tdms')
        group = ("07/09/2012 06:58:23 PM - " +
                 "Digital Input - Decimated Data_Level1")
        channel = "Dev1_port3_line7 - line 0"
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

        data = tf.object(group, channel).data
        np.testing.assert_almost_equal(data[:10], expected)

    def test_raw_format(self):
        """Test reading a file with DAQmx raw data"""
        tf = tdms.TdmsFile(_data_dir + '/raw1.tdms')
        objpath = tf.groups()[0]
        data = tf.object(objpath, 'First  Channel').data
        np.testing.assert_almost_equal(data[:10],
                                       [-0.18402661, 0.14801477, -0.24506363,
                                        -0.29725028, -0.20020142, 0.18158513,
                                        0.02380444, 0.20661031, 0.20447401,
                                        0.2517777])


if __name__ == '__main__':
    unittest.main()
