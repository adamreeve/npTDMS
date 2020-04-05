import numpy as np

from nptdms import TdmsFile
from nptdms.test.util import (
    GeneratedFile,
    hexlify_value,
    string_hexlify,
    segment_objects_metadata,
    channel_metadata)
from nptdms.test.scenarios import TDS_TYPE_INT32


def test_stream_scaled_data_chunks(benchmark):
    properties = {
        "NI_Number_Of_Scales":
            (3, "01 00 00 00"),
        "NI_Scale[0]_Scale_Type":
            (0x20, hexlify_value("<I", len("Linear")) + string_hexlify("Linear")),
        "NI_Scale[0]_Linear_Slope":
            (10, hexlify_value("<d", 2.0)),
        "NI_Scale[0]_Linear_Y_Intercept":
            (10, hexlify_value("<d", 10.0))
    }
    test_file = GeneratedFile()
    data_array = np.arange(0, 1000, dtype=np.dtype('int32'))
    data = data_array.tobytes()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 100, properties),
        ),
        data, binary_data=True
    )
    for _ in range(0, 9):
        test_file.add_segment(
            ("kTocRawData", ), "", data, binary_data=True)

    def stream_chunks(chan):
        all_data = []
        for chunk in chan.data_chunks():
            all_data.append(chunk[:])
        return all_data

    with TdmsFile.open(test_file.get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel1']
        channel_data = benchmark(stream_chunks, channel)

        channel_data = np.concatenate(channel_data)
        expected_data = np.tile(10.0 + 2.0 * data_array, 10)
        np.testing.assert_equal(channel_data, expected_data)
