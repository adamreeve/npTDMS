""" Test reading timestamp properties and data
"""

import numpy as np
from nptdms import TdmsFile
from nptdms.test.util import (
    GeneratedFile,
    channel_metadata,
    hexlify_value,
    segment_objects_metadata,
)


def test_read_raw_timestamp_properties():
    """ Test reading timestamp properties as a raw TDMS timestamp
    """
    test_file = GeneratedFile()
    second_fractions = 1234567890 * 10 ** 10
    properties = {
        "wf_start_time": (0x44, hexlify_value("<Q", second_fractions) + hexlify_value("<q", 3524551547))
    }
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2, properties),
        ),
        "01 00 00 00" "02 00 00 00"
    )

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file, raw_timestamps=True)
        start_time = tdms_data['group']['channel1'].properties['wf_start_time']
        assert start_time.seconds == 3524551547
        assert start_time.second_fractions == second_fractions
        assert start_time.as_datetime64() == np.datetime64('2015-09-08T10:05:47.669260', 'us')
        assert start_time.as_datetime64().dtype == np.dtype('datetime64[us]')
        assert start_time.as_datetime64('ns') == np.datetime64('2015-09-08T10:05:47.669260594', 'ns')
        assert start_time.as_datetime64('ns').dtype == np.dtype('datetime64[ns]')


def test_read_raw_timestamp_data():
    """ Test reading timestamp data as a raw TDMS timestamps
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 0x44, 4),
        ),
        hexlify_value("<Q", 0000000000) + hexlify_value("<q", 3672033330) +
        hexlify_value("<Q", 1234567890) + hexlify_value("<q", 3672033330) +
        hexlify_value("<Q", 0000000000) + hexlify_value("<q", 3672033331) +
        hexlify_value("<Q", 1234567890) + hexlify_value("<q", 3672033331)
    )

    expected_seconds = np.array([3672033330, 3672033331, 3672033330, 3672033331], np.dtype('int64'))
    expected_second_fractions = np.array([0, 2000000000, 0, 2000000000], np.dtype('uint64'))
    expected_timestamps = np.array([
        np.datetime64('2020-05-11 09:15:30'),
        np.datetime64('2020-05-11 09:15:30'),
        np.datetime64('2020-05-11 09:15:31'),
        np.datetime64('2020-05-11 09:15:31'),
    ])

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file, raw_timestamps=True)
        data = tdms_data['group']['channel1'][:]
        np.testing.assert_equal(data['seconds'], expected_seconds)
        np.testing.assert_equal(data['second_fractions'], expected_second_fractions)
        np.testing.assert_equal(data.as_datetime64(), expected_timestamps)


