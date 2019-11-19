"""Test TdmsSegment"""

import unittest
from datetime import datetime
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

from nptdms.writer import TdmsSegment, read_properties_dict
from nptdms.types import *


class TestObject(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class TDMSTestClass(unittest.TestCase):
    def test_write_leadin_with_one_channel(self):
        data_type = TestObject(size=4)

        channel = TestObject(
            path="",
            has_data=True,
            data=[0] * 10,
            data_type=data_type)

        toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
        metadata_size = 12

        segment = TdmsSegment([channel])
        leadin = segment.leadin(toc, metadata_size)

        expected_values = [
            Bytes(b'TDSm'),
            Int32(14),  # TOC bitmask
            Int32(4712),  # TDMS version
            Uint64(52),  # Next segment offset
            Uint64(12),  # Raw data offset
            ]

        self.assert_sequence_equal(leadin, expected_values)

    def test_write_leadin_with_object_without_data(self):
        channel = TestObject(
            path="",
            has_data=False)

        toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
        metadata_size = 12

        segment = TdmsSegment([channel])
        leadin = segment.leadin(toc, metadata_size)

        expected_values = [
            Bytes(b'TDSm'),
            Int32(14),  # TOC bitmask
            Int32(4712),  # TDMS version
            Uint64(12),  # Next segment offset
            Uint64(12),  # Raw data offset
            ]

        self.assert_sequence_equal(leadin, expected_values)

    def test_write_metadata_with_properties(self):
        data_type = TestObject(enum_value=3)

        # Use an ordered dict for properties so that
        # the order of properties in metadata is guaranteed
        properties = OrderedDict()
        properties["prop1"] = String("foo")
        properties["prop2"] = Int32(42)

        channel = TestObject(
            path="channel_path",
            has_data=True,
            data=[1] * 10,
            data_type=data_type,
            properties=properties)

        segment = TdmsSegment([channel])
        metadata = segment.metadata()

        expected_values = [
            Uint32(1),  # Number of objects
            String("channel_path"),
            Uint32(20),  # Length of raw data index in bytes
            Int32(3),  # Data type
            Uint32(1),  # Array dimension
            Uint64(10),  # Number of values
            Uint32(2),  # Number of properties
            String("prop1"),  # Property name
            Int32(0x20),
            String("foo"),
            String("prop2"),
            Int32(3),
            Int32(42),
            ]

        self.assert_sequence_equal(metadata, expected_values)

    def test_write_metadata_with_no_data(self):
        obj = TestObject(
            path="object_path",
            has_data=False,
            properties={})

        segment = TdmsSegment([obj])
        metadata = segment.metadata()

        expected_values = [
            Uint32(1),  # Number of objects
            String("object_path"),
            Bytes(b'\xFF\xFF\xFF\xFF'),  # Raw data index
            Uint32(0),  # Number of properties
            ]

        self.assert_sequence_equal(metadata, expected_values)

    def test_properties_are_converted_to_tdms_types(self):
        test_time = datetime.utcnow()

        properties = {
            "prop1": Int32(1),
            "prop2": 2,
            "prop3": "foo",
            "prop4": True,
            "prop5": 3.142,
            "prop6": test_time,
        }

        tdms_properties = read_properties_dict(properties)

        self.assertEqual(len(tdms_properties), len(properties))
        self.assertEqual(tdms_properties["prop1"], Int32(1))
        self.assertEqual(tdms_properties["prop2"], Int32(2))
        self.assertEqual(tdms_properties["prop3"], String("foo"))
        self.assertEqual(tdms_properties["prop4"], Boolean(True))
        self.assertEqual(tdms_properties["prop5"], DoubleFloat(3.142))
        self.assertEqual(tdms_properties["prop6"], TimeStamp(test_time))

    def test_datetime_converted_when_it_only_has_date_part(self):
        test_time = np.datetime64('2017-11-19')

        properties = {
            "time_prop": test_time,
        }

        tdms_properties = read_properties_dict(properties)

        self.assertEqual(tdms_properties["time_prop"], TimeStamp(test_time))

    def test_writing_long_integer_properties(self):
        properties = {
            "prop1": 2147483647,
            "prop2": 2147483648,
        }

        tdms_properties = read_properties_dict(properties)

        self.assertEqual(len(tdms_properties), len(properties))
        self.assertEqual(tdms_properties["prop1"], Int32(2147483647))
        self.assertEqual(tdms_properties["prop2"], Int64(2147483648))

    def test_error_raised_when_cannot_convert_property_value(self):
        properties = {
            "prop1": None
        }

        with self.assertRaises(TypeError):
            tdms_properties = read_properties_dict(properties)

    def assert_sequence_equal(self, values, expected_values):
        position = 1
        expected_values = iter(expected_values)
        for val in values:
            try:
                expected = next(expected_values)
            except StopIteration:
                raise ValueError(
                    "Expected end of sequence at position %d but found: %r" %
                    (position, val))
            self.assertEqual(
                val, expected,
                msg="Expected %r to equal %r at position %d" %
                (val, expected, position))
            position += 1
        try:
            expected = next(expected_values)
            raise ValueError(
                "Expected %r at position %d but found end of sequence" %
                (expected, position))
        except StopIteration:
            pass
