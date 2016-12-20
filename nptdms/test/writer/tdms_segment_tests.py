"""Test TdmsSegment"""

import unittest

from nptdms.writer import TdmsSegment
from nptdms.value import Bytes, Int32, Uint64


class TestObject(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class TDMSTestClass(unittest.TestCase):
    def test_write_leadin_with_one_channel(self):
        data_type = TestObject(length=4)

        channel = TestObject(
            path=lambda: "",
            has_data=lambda: True,
            data=[0] * 10,
            data_type=lambda: data_type)

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
            path=lambda: "",
            has_data=lambda: False)

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

    def assert_sequence_equal(self, values, expected_values):
        expected_values = iter(expected_values)
        for val in values:
            try:
                expected = next(expected_values)
            except StopIteration:
                raise ValueError(
                    "Expected end of sequence but found: %r" % val)
            self.assertEqual(val, expected)
        try:
            expected = next(expected_values)
            raise ValueError(
                "Expected %r but found end of sequence" % expected)
        except StopIteration:
            pass
