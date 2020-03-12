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


def segment_objects_metadata(*args):
    """ Metadata for multiple objects in a segment
    """
    num_objects_hex = hexlify_value("<I", len(args))
    return num_objects_hex + "".join(args)


def channel_metadata(channel_name, data_type, num_values, properties=None):
    return (
        # Length of the object path
        hexlify_value('<I', len(channel_name)) +
        # Object path
        string_hexlify(channel_name) +
        # Length of index information
        "14 00 00 00" +
        # Raw data data type
        hexlify_value('<I', data_type) +
        # Dimension
        "01 00 00 00" +
        # Number of raw data values
        hexlify_value('<Q', num_values) +
        hex_properties(properties)
    )


def hex_properties(properties):
    if properties is None:
        properties = {}
    props_hex = hexlify_value('<I', len(properties))
    for (prop_name, (prop_type, prop_value)) in properties.items():
        props_hex += hexlify_value('<I', len(prop_name))
        props_hex += string_hexlify(prop_name)
        props_hex += hexlify_value('<I', prop_type)
        props_hex += prop_value
    return props_hex


def channel_metadata_with_repeated_structure(channel_name):
    return (
        # Length of the object path
        hexlify_value('<I', len(channel_name)) +
        # Object path
        string_hexlify(channel_name) +
        # Raw data index header meaning repeat previous data structure
        "00 00 00 00" +
        # Number of properties (0)
        "00 00 00 00"
    )


def channel_metadata_with_no_data(channel_name):
    return (
        # Length of the object path
        hexlify_value('<I', len(channel_name)) +
        # Object path
        string_hexlify(channel_name) +
        # Raw data index header meaning no data in this segment
        "FF FF FF FF" +
        # Number of properties (0)
        "00 00 00 00"
    )


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
        # Number of raw data values
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
        "01 00 00 00"
        # Length of property name
        "03 00 00 00"
        # Property name (num)
        "6E 75 6D"
        # Data type of property
        "03 00 00 00"
        # Value
        "0F 00 00 00"
    )
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "03 00 00 00"
        "04 00 00 00"
    )
    return toc, metadata, data


class GeneratedFile(object):
    """Generate a TDMS file for testing"""

    def __init__(self):
        self._content = b''

    def add_segment(self, toc, metadata, data, incomplete=False):
        metadata_bytes = _hex_to_bytes(metadata)
        data_bytes = _hex_to_bytes(data)
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
            lead_in += _hex_to_bytes("69 12 00 00")
            next_segment_offset = len(metadata_bytes) + len(data_bytes)
            raw_data_offset = len(metadata_bytes)
            if incomplete:
                lead_in += _hex_to_bytes('FF' * 8)
            else:
                lead_in += struct.pack('<Q', next_segment_offset)
            lead_in += struct.pack('<Q', raw_data_offset)
        else:
            lead_in = b''
        self._content += lead_in + metadata_bytes + data_bytes

    def get_tempfile(self, **kwargs):
        named_file = tempfile.NamedTemporaryFile(suffix=".tdms", **kwargs)
        file = named_file.file
        file.write(self._content)
        file.seek(0)
        return named_file

    def load(self, *args, **kwargs):
        with tempfile.NamedTemporaryFile(suffix=".tdms") as named_file:
            file = named_file.file
            file.write(self._content)
            file.seek(0)
            return tdms.TdmsFile(file, *args, **kwargs)


class BytesIoTestFile(GeneratedFile):
    def load(self, *args, **kwargs):
        file = BytesIO()
        file.write(self._content)
        file.seek(0)
        return tdms.TdmsFile(file, *args, **kwargs)


def _hex_to_bytes(hex_data):
    """ Converts a string of hex to a byte array
    """
    return binascii.unhexlify(
        hex_data.replace(" ", "").replace("\n", "").encode('utf-8'))
