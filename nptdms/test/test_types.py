"""Test type reading and writing"""

from datetime import date, datetime
import unittest
import io
import numpy as np

from nptdms import types


class TimeStampTests(unittest.TestCase):
    def test_timestamp_roundtrip(self):
        """Test timestamp write from datetime64 then read"""

        self._roundtrip_test('2019-11-08T18:47:00')

    def test_roundtrip_before_epoch(self):
        """Test reading timestamp before TDMS epoch"""

        self._roundtrip_test('0000-01-01T05:00:00')

    def test_timestamp_with_microseconds(self):
        """Test timestamp roundtrip with microseconds"""

        self._roundtrip_test('2019-11-08T18:47:00.123456')

    def test_roundtrip_before_epoch_with_microseconds(self):
        """Test reading timestamp before TDMS epoch with microseconds"""

        self._roundtrip_test('1903-12-31T23:59:59.500')

    def test_timestamp_from_datetime(self):
        """Test timestamp from built in datetime value"""

        input_datetime = datetime(2019, 11, 8, 18, 47, 0)
        expected_datetime = np.datetime64('2019-11-08T18:47:00')

        timestamp = types.TimeStamp(input_datetime)
        data_file = io.BytesIO(timestamp.bytes)

        read_datetime = types.TimeStamp.read(data_file)

        self.assertEqual(expected_datetime, read_datetime)

    def test_timestamp_from_date(self):
        """Test timestamp from built in date value"""

        input_datetime = date(2019, 11, 8)
        expected_datetime = np.datetime64('2019-11-08T00:00:00')

        timestamp = types.TimeStamp(input_datetime)
        data_file = io.BytesIO(timestamp.bytes)

        read_datetime = types.TimeStamp.read(data_file)

        self.assertEqual(expected_datetime, read_datetime)

    def _roundtrip_test(self, time_string):
        expected_datetime = np.datetime64(time_string)

        timestamp = types.TimeStamp(expected_datetime)
        data_file = io.BytesIO(timestamp.bytes)

        read_datetime = types.TimeStamp.read(data_file)

        self.assertEqual(expected_datetime, read_datetime)


if __name__ == '__main__':
    unittest.main()
