import numpy as np

from nptdms import types
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class DaqMxMetadata(object):
    """ Describes DAQmx data
    """

    __slots__ = [
        'chunk_size',
        'data_type',
        'dimension',
        'raw_data_widths',
        'scale_id',
        'scaler_data_type',
        'scaler_data_type_code',
        'scaler_raw_buffer_index',
        'scaler_raw_buffer_index',
        'scaler_raw_byte_offset',
        'scaler_sample_format_bitmap',
        'scaler_vector_length',
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
        self.scaler_vector_length = types.Uint32.read(f, endianness)
        # Size of the vector
        log.debug("mxDAQ format scaler vector size '%d'",
                  self.scaler_vector_length)
        if self.scaler_vector_length > 1:
            log.error("mxDAQ multiple format changing scalers not implemented")

        for _ in range(self.scaler_vector_length):
            # WARNING: This code overwrites previous values with new
            # values.  At this time NI provides no documentation on
            # how to use these scalers and sample TDMS files do not
            # include more than one of these scalers.
            self.scaler_data_type_code = types.Uint32.read(f, endianness)
            self.scaler_data_type = (
                types.tds_data_types[self.scaler_data_type_code])

            # more info for format changing scaler
            self.scaler_raw_buffer_index = types.Uint32.read(f, endianness)
            self.scaler_raw_byte_offset = types.Uint32.read(f, endianness)
            self.scaler_sample_format_bitmap = types.Uint32.read(f, endianness)
            self.scale_id = types.Uint32.read(f, endianness)

        raw_data_widths_length = types.Uint32.read(f, endianness)
        self.raw_data_widths = np.zeros(raw_data_widths_length, dtype=np.int32)
        for cnt in range(raw_data_widths_length):
            self.raw_data_widths[cnt] = types.Uint32.read(f, endianness)

    def __str__(self):
        """ Return string representation of DAQmx metadata
        """
        properties = (
            "%s: %s" % (name, getattr(self, name))
            for name in self.__slots__)

        properties_list = ", ".join(properties)
        return "%s: ('%s')" % (self.__class__.__name__, properties_list)
