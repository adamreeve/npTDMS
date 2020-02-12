""" Utilities for testing TDMS reading
"""

import binascii
from io import BytesIO
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


class BytesIoTestFile(TestFile):
    def __init__(self):
        self.file = BytesIO()
        self.data = bytes()
