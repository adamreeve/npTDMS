from collections import defaultdict
import numpy as np

from nptdms import types
from nptdms.base_segment import (
    BaseSegment, BaseSegmentObject, DataChunk, read_interleaved_segment_bytes)
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class DaqmxSegment(BaseSegment):
    """ A TDMS segment with DAQmx data
    """

    def _new_segment_object(self, object_path):
        return DaqmxSegmentObject(object_path, self.endianness)

    def _get_chunk_size(self):
        # For DAQmxRawData, each channel in a segment has the same number
        # of values and contains the same raw data widths, so use
        # the first valid channel metadata to calculate the data size.
        try:
            return next(
                o.number_values * o.total_raw_data_width
                for o in self.ordered_objects
                if o.has_data and
                o.number_values * o.total_raw_data_width > 0)
        except StopIteration:
            return 0

    def _read_data_chunk(self, file, data_objects, chunk_index):
        """Read data from DAQmx data segment"""

        log.debug("Reading DAQmx data segment")

        all_daqmx = all(
            o.data_type == types.DaqMxRawData for o in data_objects)
        if not all_daqmx:
            raise Exception("Cannot read a mix of DAQmx and "
                            "non-DAQmx interleaved data")

        # If we have DAQmx data, we expect all objects to have matching
        # raw data widths, so just use the first object:
        raw_data_widths = data_objects[0].daqmx_metadata.raw_data_widths
        chunk_size = data_objects[0].number_values
        scaler_data = defaultdict(dict)

        # Data for each set of raw data (corresponding to one card) is
        # interleaved separately, so read one after another
        for (raw_buffer_index, raw_data_width) in enumerate(raw_data_widths):
            # Read all data into 1 byte unsigned ints first
            combined_data = read_interleaved_segment_bytes(
                file, raw_data_width, chunk_size)

            # Now set arrays for each scaler of each channel where the scaler
            # data comes from this set of raw data
            for (i, obj) in enumerate(data_objects):
                scalers_for_raw_buffer_index = [
                    scaler for scaler in obj.daqmx_metadata.scalers
                    if scaler.raw_buffer_index == raw_buffer_index]
                for scaler in scalers_for_raw_buffer_index:
                    offset = scaler.raw_byte_offset
                    scaler_size = scaler.data_type.size
                    byte_columns = tuple(
                        range(offset, offset + scaler_size))
                    log.debug("Byte columns for channel %d scaler %d: %s",
                              i, scaler.scale_id, byte_columns)
                    # Select columns for this scaler, so that number of values
                    # will be number of bytes per point * number of data
                    # points. Then use ravel to flatten the results into a
                    # vector.
                    this_scaler_data = combined_data[:, byte_columns].ravel()
                    # Now set correct data type, so that the array length
                    # should be correct
                    this_scaler_data.dtype = (
                        scaler.data_type.nptype.newbyteorder(self.endianness))
                    scaler_data[obj.path][scaler.scale_id] = this_scaler_data

        return DataChunk.scaler_data(scaler_data)


class DaqmxSegmentObject(BaseSegmentObject):
    """ A DAQmx TDMS segment object
    """

    __slots__ = ['daqmx_metadata']

    def __init__(self, path, endianness):
        super(DaqmxSegmentObject, self).__init__(path, endianness)
        self.daqmx_metadata = None

    def read_raw_data_index(self, f, raw_data_index_header):
        if raw_data_index_header not in (0x00001269, 0x00001369):
            raise ValueError(
                "Unexpected raw data index for DAQmx data: 0x%08X" %
                raw_data_index_header)
        # This is a DAQmx raw data segment.
        #    0x00001269 for segment containing Format Changing scaler.
        #    0x00001369 for segment containing Digital Line scaler.
        if raw_data_index_header == 0x00001369:
            # special scaling for DAQ's digital input lines?
            log.warning("DAQmx with Digital Line scaler has not tested")

        # Read the data type
        data_type_val = types.Uint32.read(f, self.endianness)
        try:
            self.data_type = types.tds_data_types[data_type_val]
        except KeyError:
            raise KeyError("Unrecognised data type: %s" % data_type_val)

        log.debug("DAQmx object data type: %r", self.data_type)

        daqmx_metadata = DaqMxMetadata(f, self.endianness)
        log.debug("DAQmx metadata: %r", daqmx_metadata)

        self.data_type = daqmx_metadata.data_type
        # DAQmx format has special chunking
        self.data_size = daqmx_metadata.chunk_size * sum(daqmx_metadata.raw_data_widths)
        self.number_values = daqmx_metadata.chunk_size
        self.daqmx_metadata = daqmx_metadata

    @property
    def total_raw_data_width(self):
        return sum(self.daqmx_metadata.raw_data_widths)

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


# Type codes for DAQmx scalers don't match the normal TDMS type codes:
DAQMX_TYPES = {
    0: types.Uint8,
    1: types.Int8,
    2: types.Uint16,
    3: types.Int16,
    4: types.Uint32,
    5: types.Int32,
}
