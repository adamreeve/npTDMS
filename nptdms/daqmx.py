import numpy as np

from nptdms import types
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class DaqMxMetadata(object):
    """ Describes DAQmx data
    """

    __slots__ = [
        'data_type',
        'dimension',
        'chunk_size',
        'raw_data_widths',
        'scalers',
        ]

    def __init__(self, f, endianness):
        """
        Read the metadata for a DAQmx raw segment.  This is the raw
        DAQmx-specific portion of the raw data index.
        """
        self.data_type = types.tds_data_types[0xFFFFFFFF]
        self.dimension = types.Uint32.read(f, endianness)
        # In TDMS format version 2.0, 1 is the only valid value for dimension
        if self.dimension != 1:
            log.warning("Data dimension is not 1")
        self.chunk_size = types.Uint64.read(f, endianness)

        # size of vector of format changing scalers
        scaler_vector_length = types.Uint32.read(f, endianness)
        log.debug("mxDAQ format scaler vector size '%d'", scaler_vector_length)
        self.scalers = [
            DaqMxScaler(f, endianness)
            for _ in range(scaler_vector_length)]

        # Read raw data widths.
        # This is an array of widths in bytes, which should be the same
        # for all channels that have DAQmx data. It's unclear what it means
        # when there are multiple entries in this array.
        raw_data_widths_length = types.Uint32.read(f, endianness)
        self.raw_data_widths = np.zeros(raw_data_widths_length, dtype=np.int32)
        for width_idx in range(raw_data_widths_length):
            self.raw_data_widths[width_idx] = types.Uint32.read(f, endianness)

    def __repr__(self):
        """ Return string representation of DAQmx metadata
        """
        properties = (
            "%s=%s" % (name, getattr(self, name))
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
        self.sample_format_bitmap = types.Uint32.read(
            open_file, endianness)
        self.scale_id = types.Uint32.read(open_file, endianness)

    def __repr__(self):
        properties = (
            "%s=%s" % (name, getattr(self, name))
            for name in self.__slots__)

        properties_list = ", ".join(properties)
        return "%s(%s)" % (self.__class__.__name__, properties_list)


# Type codes for DAQmx scalers don't appear to match
# the  normal TDMS type codes:
DAQMX_TYPES = {
    0: types.Uint8,
    1: types.Int8,
    2: types.Uint16,
    3: types.Int16,
    4: types.Uint32,
    5: types.Int32,
}
