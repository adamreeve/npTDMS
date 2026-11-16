"""Numeric channel arrays must match the byte order of written TDMS segments."""

from io import BytesIO
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from nptdms import ChannelObject, TdmsFile, TdmsWriter, types


@pytest.mark.parametrize('byte_order', ['<', '>'])
@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f4', 'f8', 'c8', 'c16', '?'])
def test_numeric_channel_values_preserve_byte_order(dtype, byte_order):
    dtype = np.dtype(byte_order + dtype)
    values = {
        'i': [-2, 0, 12], 'u': [1, 128, 255], 'f': [1.25, -2.5, np.inf, np.nan],
        'c': [1 + 2j, -3 + 4j], 'b': [True, False],
    }[dtype.kind]
    data = np.array(values, dtype=dtype)
    original_bytes = data.tobytes()
    output = BytesIO()
    with TdmsWriter(output) as writer:
        writer.write_segment(ChannelObject('g', 'a', data))
    actual = TdmsFile.read(BytesIO(output.getvalue()))['g']['a'][:]
    np.testing.assert_array_equal(actual, data)
    assert actual.dtype.newbyteorder('=') == data.dtype.newbyteorder('=')
    assert data.dtype == dtype
    assert data.tobytes() == original_bytes


@pytest.mark.parametrize('dtype', ['>i2', '>u2', '>i4', '>u4', '>i8', '>u8', '>f4', '>f8', '>c8', '>c16'])
def test_empty_big_endian_channel_keeps_numeric_type(dtype):
    output = BytesIO()
    data = np.array([], dtype=dtype)
    with TdmsWriter(output) as writer:
        writer.write_segment(ChannelObject('g', 'a', data))
    actual = TdmsFile.read(BytesIO(output.getvalue()))['g']['a'][:]
    assert len(actual) == 0
    assert actual.dtype.newbyteorder('=') == data.dtype.newbyteorder('=')


@pytest.mark.parametrize('dtype', ['<i4', '>i4', '<f8', '>f8', '<c16', '>c16'])
def test_strided_read_only_array_is_not_modified(dtype):
    base = np.arange(8).astype(dtype)
    data = base[::-2]
    data.flags.writeable = False
    original = base.tobytes()
    output = BytesIO()
    with TdmsWriter(output) as writer:
        writer.write_segment(ChannelObject('g', 'a', data))
    np.testing.assert_array_equal(TdmsFile.read(BytesIO(output.getvalue()))['g']['a'][:], data)
    assert base.tobytes() == original
    assert not data.flags.writeable
    assert data.dtype == np.dtype(dtype)


@pytest.mark.parametrize('version', [4712, 4713])
def test_mixed_endian_segments_with_index_file(tmp_path, version):
    path = str(tmp_path / 'mixed.tdms')
    with TdmsWriter(path, version=version, index_file=True) as writer:
        writer.write_segment(ChannelObject('g', 'a', np.array([1, 2], dtype='>i2')),
                             ChannelObject('g', 'b', np.array([1.25, 2.5], dtype='<f8')))
        writer.write_segment(ChannelObject('g', 'a', np.array([3, 4], dtype='<i2')),
                             ChannelObject('g', 'b', np.array([3.5, 4.25], dtype='>f8')))
    with TdmsFile.open(path) as actual:
        assert actual.tdms_version == version
        np.testing.assert_array_equal(actual['g']['a'][1:3], [2, 3])
        np.testing.assert_array_equal(actual['g']['b'][1:3], [2.5, 3.5])


def test_big_endian_payload_is_written_as_little_endian():
    output = BytesIO()
    with TdmsWriter(output) as writer:
        writer.write_segment(ChannelObject('g', 'a', np.array([0x0102, -3], dtype='>i2')))
    contents = output.getvalue()
    toc_mask, = struct.unpack_from('<I', contents, 4)
    metadata_size, = struct.unpack_from('<Q', contents, 20)
    assert toc_mask & 0x40 == 0
    assert contents[28 + metadata_size:] == b'\x02\x01\xfd\xff'


def test_numeric_object_with_declared_type_uses_little_endian_payload():
    data = np.array([0x0102, -3], dtype='>i2')
    channel = SimpleNamespace(path="/'g'/'a'", data=data, data_type=types.Int16, properties={})
    output = BytesIO()
    with TdmsWriter(output) as writer:
        writer.write_segment([channel])
    np.testing.assert_array_equal(TdmsFile.read(BytesIO(output.getvalue()))['g']['a'][:], data)


def test_copy_channel_from_big_endian_tdms_file():
    def object_metadata(path, index):
        encoded = path.encode('utf-8')
        return struct.pack('>I', len(encoded)) + encoded + index + struct.pack('>I', 0)

    no_data = struct.pack('>I', 0xFFFFFFFF)
    metadata = struct.pack('>I', 3)
    metadata += object_metadata('/', no_data)
    metadata += object_metadata("/'g'", no_data)
    metadata += object_metadata("/'g'/'a'", struct.pack('>IIIQ', 20, 2, 1, 2))
    payload = struct.pack('>hh', 0x0102, -3)
    contents = b'TDSm' + struct.pack('<I', 0x4E)
    contents += struct.pack('>IQQ', 4712, len(metadata) + len(payload), len(metadata))
    source = TdmsFile.read(BytesIO(contents + metadata + payload))
    np.testing.assert_array_equal(source['g']['a'][:], [0x0102, -3])
    output = BytesIO()
    with TdmsWriter(output) as writer:
        writer.write_segment([source['g']['a']])
    np.testing.assert_array_equal(TdmsFile.read(BytesIO(output.getvalue()))['g']['a'][:], [0x0102, -3])
