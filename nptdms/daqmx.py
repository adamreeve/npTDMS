from collections import defaultdict
import numpy as np

from nptdms import types
from nptdms.base_segment import (
    BaseSegmentObject, BaseDataReader, RawDataChunk, read_interleaved_segment_bytes)
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)

FORMAT_CHANGING_SCALER = 0x00001269
DIGITAL_LINE_SCALER = 0x0000126A


class DaqmxDataReader(BaseDataReader):
    """ A TDMS segment with DAQmx data
    """
    def _read_data_chunk(self, file, data_objects, chunk_index):
        """Read data from DAQmx data segment"""

        log.debug("Reading DAQmx data segment")

        all_daqmx = all(
            o.data_type == types.DaqMxRawData for o in data_objects)
        if not all_daqmx:
            raise Exception("Cannot read a mix of DAQmx and "
                            "non-DAQmx interleaved data")

        scaler_data = defaultdict(dict)

        # Data for each raw data buffer is interleaved separately, so read one after another
        for (raw_buffer_index, buffer_shape) in enumerate(get_buffer_dimensions(data_objects)):
            (chunk_size, raw_data_width) = buffer_shape
            # Read all data into 1 byte unsigned ints first
            combined_data = read_interleaved_segment_bytes(file, raw_data_width, chunk_size)

            # Now get arrays for each scaler of each channel where the scaler
            # data comes from this raw buffer
            for (i, obj) in enumerate(data_objects):
                scalers_for_raw_buffer_index = [
                    scaler for scaler in obj.daqmx_metadata.scalers
                    if scaler.raw_buffer_index == raw_buffer_index]
                for scaler in scalers_for_raw_buffer_index:
                    byte_offset = scaler.byte_offset()
                    scaler_size = scaler.data_type.size
                    byte_columns = tuple(
                        range(byte_offset, byte_offset + scaler_size))
                    # Select columns for this scaler, so that number of values
                    # will be number of bytes per point * number of data points.
                    # Then use ravel to flatten the results into a vector.
                    this_scaler_data = combined_data[:, byte_columns].ravel()
                    # Now set correct data type, so that the array length
                    # should be correct
                    this_scaler_data.dtype = (
                        scaler.data_type.nptype.newbyteorder(self.endianness))
                    processed_data = scaler.postprocess_data(this_scaler_data)
                    scaler_data[obj.path][scaler.scale_id] = processed_data

        return RawDataChunk.scaler_data(scaler_data)


def get_daqmx_chunk_size(ordered_objects):
    # For DAQmx data, each channel should specify the same raw data widths,
    # but different buffers may have different numbers of values.
    return sum((num_values * width) for (num_values, width) in get_buffer_dimensions(ordered_objects))


def get_buffer_dimensions(ordered_objects):
    """ Returns DAQmx buffer dimensions as list of tuples of (number of values, width in bytes)
    """
    dimensions = None
    for o in ordered_objects:
        if not o.has_data:
            continue
        daqmx_metadata = o.daqmx_metadata
        if dimensions is None:
            # Set width for each buffer
            dimensions = [(0, w) for w in daqmx_metadata.raw_data_widths]
        else:
            # Verify raw data widths matches previous channels
            assert len(daqmx_metadata.raw_data_widths) == len(dimensions)
            for i, dim in enumerate(dimensions):
                assert daqmx_metadata.raw_data_widths[i] == dimensions[i][1]
        # Now set the buffer number of values based on the object chunk size
        for scaler in daqmx_metadata.scalers:
            buffer_index = scaler.raw_buffer_index
            current_buffer_shape = dimensions[buffer_index]
            updated_num_values = max(current_buffer_shape[0], o.number_values)
            dimensions[buffer_index] = (updated_num_values, current_buffer_shape[1])

    return [] if dimensions is None else dimensions


class DaqmxSegmentObject(BaseSegmentObject):
    """ A DAQmx TDMS segment object
    """

    __slots__ = ['daqmx_metadata']

    def __init__(self, path, endianness):
        super(DaqmxSegmentObject, self).__init__(path, endianness)
        self.daqmx_metadata = None

    def read_raw_data_index(self, f, raw_data_index_header):
        if raw_data_index_header not in (FORMAT_CHANGING_SCALER, DIGITAL_LINE_SCALER):
            raise ValueError(
                "Unexpected raw data index for DAQmx data: 0x%08X" %
                raw_data_index_header)
        # This is a DAQmx raw data segment.
        #    0x00001269 for segment containing Format Changing scaler.
        #    0x0000126A for segment containing Digital Line scaler.
        # Note that the NI docs on the TDMS format state that digital line scaler data
        # has 0x00001369, which appears to be incorrect

        # Read the data type
        data_type_val = types.Uint32.read(f, self.endianness)
        try:
            self.data_type = types.tds_data_types[data_type_val]
        except KeyError:
            raise KeyError("Unrecognised data type: %s" % data_type_val)

        daqmx_metadata = DaqMxMetadata(f, self.endianness, raw_data_index_header)
        log.debug("DAQmx metadata: %r", daqmx_metadata)

        # DAQmx format has special chunking
        self.number_values = daqmx_metadata.chunk_size
        self.daqmx_metadata = daqmx_metadata

    @property
    def scaler_data_types(self):
        if self.daqmx_metadata is None:
            return None
        return dict(
            (s.scale_id, s.data_type)
            for s in self.daqmx_metadata.scalers)


class DaqMxMetadata(object):
    """ Describes DAQmx data for a single channel
    """

    __slots__ = [
        'scaler_type',
        'dimension',
        'chunk_size',
        'raw_data_widths',
        'scalers',
        ]

    def __init__(self, f, endianness, scaler_type):
        """
        Read the metadata for a DAQmx raw segment.  This is the raw
        DAQmx-specific portion of the raw data index.
        """
        self.scaler_type = scaler_type
        self.dimension = types.Uint32.read(f, endianness)
        # In TDMS format version 2.0, 1 is the only valid value for dimension
        if self.dimension != 1:
            raise ValueError("Data dimension is not 1")
        self.chunk_size = types.Uint64.read(f, endianness)

        # size of vector of format changing scalers
        scaler_vector_length = types.Uint32.read(f, endianness)
        scaler_class = _scaler_classes[scaler_type]
        self.scalers = [
            scaler_class(f, endianness)
            for _ in range(scaler_vector_length)]

        # Read raw data widths.
        # This is an array of widths in bytes, which should be the same
        # for all channels that have DAQmx data in a segment.
        # There is one element per acquisition card, as data is interleaved
        # separately for each card.
        raw_data_widths_length = types.Uint32.read(f, endianness)
        self.raw_data_widths = np.zeros(raw_data_widths_length, dtype=np.int32)
        for width_idx in range(raw_data_widths_length):
            self.raw_data_widths[width_idx] = types.Uint32.read(f, endianness)

    def __repr__(self):
        """ Return string representation of DAQmx metadata
        """
        properties = (
            "%s=%s" % (name, _get_attr_repr(self, name))
            for name in self.__slots__)

        properties_list = ", ".join(properties)
        return "%s(%s)" % (self.__class__.__name__, properties_list)


class DaqMxScaler(object):
    """ Details of a DAQmx raw data scaler read from a TDMS file
    """

    __slots__ = [
        'scale_id',
        'data_type',
        'raw_buffer_index',
        'raw_byte_offset',
        'sample_format_bitmap',
        ]

    def __init__(self, open_file, endianness):
        data_type_code = types.Uint32.read(open_file, endianness)
        self.data_type = DAQMX_TYPES[data_type_code]

        # more info for format changing scaler
        self.raw_buffer_index = types.Uint32.read(open_file, endianness)
        self.raw_byte_offset = types.Uint32.read(open_file, endianness)
        self.sample_format_bitmap = types.Uint32.read(open_file, endianness)
        self.scale_id = types.Uint32.read(open_file, endianness)

    def byte_offset(self):
        return self.raw_byte_offset

    def postprocess_data(self, data):
        return data

    def __repr__(self):
        properties = (
            "%s=%s" % (name, _get_attr_repr(self, name))
            for name in self.__slots__)

        properties_list = ", ".join(properties)
        return "%s(%s)" % (self.__class__.__name__, properties_list)


class DigitalLineScaler(object):
    """ Details of a DAQmx digital line scaler read from a TDMS file
    """

    __slots__ = [
        'scale_id',
        'data_type',
        'raw_buffer_index',
        'raw_bit_offset',
        'sample_format_bitmap',
        ]

    def __init__(self, open_file, endianness):
        data_type_code = types.Uint32.read(open_file, endianness)
        self.data_type = DAQMX_TYPES[data_type_code]

        # more info for digital line scaler
        self.raw_buffer_index = types.Uint32.read(open_file, endianness)
        self.raw_bit_offset = types.Uint32.read(open_file, endianness)
        self.sample_format_bitmap = types.Uint8.read(open_file, endianness)
        self.scale_id = types.Uint32.read(open_file, endianness)

    def byte_offset(self):
        return self.raw_bit_offset // 8

    def postprocess_data(self, data):
        bit_offset = self.raw_bit_offset % 8
        bitmask = 1 << bit_offset
        return np.right_shift(np.bitwise_and(data, bitmask), bit_offset)

    def __repr__(self):
        properties = (
            "%s=%s" % (name, _get_attr_repr(self, name))
            for name in self.__slots__)

        properties_list = ", ".join(properties)
        return "%s(%s)" % (self.__class__.__name__, properties_list)


def _get_attr_repr(obj, attr_name):
    val = getattr(obj, attr_name)
    if isinstance(val, type):
        return val.__name__
    return repr(val)


# Type codes for DAQmx scalers don't match the normal TDMS type codes:
DAQMX_TYPES = {
    0: types.Uint8,
    1: types.Int8,
    2: types.Uint16,
    3: types.Int16,
    4: types.Uint32,
    5: types.Int32,
}


_scaler_classes = {
    FORMAT_CHANGING_SCALER: DaqMxScaler,
    DIGITAL_LINE_SCALER: DigitalLineScaler,
}
