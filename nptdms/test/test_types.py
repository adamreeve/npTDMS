"""Test type reading and writing"""

from datetime import date, datetime
import io
import numpy as np
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

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime


def test_timestamp_from_datetime():
    """Test timestamp from built in datetime value"""

    input_datetime = datetime(2019, 11, 8, 18, 47, 0)
    expected_datetime = np.datetime64('2019-11-08T18:47:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime


def test_timestamp_from_date():
    """Test timestamp from built in date value"""

    input_datetime = date(2019, 11, 8)
    expected_datetime = np.datetime64('2019-11-08T00:00:00')

    timestamp = types.TimeStamp(input_datetime)
    data_file = io.BytesIO(timestamp.bytes)

    read_datetime = types.TimeStamp.read(data_file)

    assert expected_datetime == read_datetime
