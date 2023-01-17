"""Test type reading and writing"""

from datetime import date, datetime
import io
import numpy as np
import struct
import pytest

from nptdms import types


@pytest.mark.parametrize(
    "time_string",
    [
        pytest.param('2019-11-08T18:47:00', id="standard timestamp"),
        pytest.param('0000-01-01T05:00:00', id="timestamp before TDMS epoch"),
        pytest.param('2019-11-08T18:47:00.123456', id="timestamp with microseconds"),
        pytest.param('1903-12-31T23:59:59.500', id="timestamp before TDMS epoch with microseconds"),
    ]
)
def test_timestamp_round_trip(time_string):
    expected_datetime = np.datetime64(time_string)

    timestamp = types.TimeStamp(expected_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file).as_datetime64()

    assert expected_datetime == read_datetime


def test_timestamp_from_datetime():
    """Test timestamp from built in datetime value"""

    input_datetime = datetime(2019, 11, 8, 18, 47, 0)
    expected_datetime = np.datetime64('2019-11-08T18:47:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime.as_datetime64()


def test_timestamp_from_date():
    """Test timestamp from built in date value"""

    input_datetime = date(2019, 11, 8)
    expected_datetime = np.datetime64('2019-11-08T00:00:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime.as_datetime64()


def test_invalid_utf8_string_read(caplog):
    """ Test reading a single invalid string value"""
    file = io.BytesIO(struct.pack("<L", 3) + b'0 \xb0')
    string_value = types.String.read(file)

    assert string_value == "0 �"
    assert "WARNING" in caplog.text
    assert "0 \\xb0" in caplog.text


def test_invalid_utf8_strings_read(caplog):
    """Test reading multiple string values where one is invalid"""
    string_bytes = [
        b'hello',
        b'0 \xb0',
        b'world',
    ]
    offset = 0
    offsets = []
    for val in string_bytes:
        offset += len(val)
        offsets.append(struct.pack("<L", offset))
    file = io.BytesIO(b''.join(offsets + string_bytes))
    string_values = types.String.read_values(file, len(string_bytes))

    assert string_values == ["hello", "0 �", "world"]
    assert "WARNING" in caplog.text
    assert "0 \\xb0" in caplog.text
