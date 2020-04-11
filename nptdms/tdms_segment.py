import os
import numpy as np

from nptdms import types
from nptdms.base_segment import (
    BaseSegment,
    BaseSegmentObject,
    RawChannelDataChunk,
    RawDataChunk,
    read_interleaved_segment_bytes,
    fromfile)
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class InterleavedDataSegment(BaseSegment):
    """ A TDMS segment with interleaved data
    """

    __slots__ = []

    def _new_segment_object(self, object_path):
        return TdmsSegmentObject(object_path, self.endianness)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        # If all data types are sized and all the lengths are
        # the same, then we can read all data at once with numpy,
        # which is much faster
        all_sized = all(
            o.data_type.size is not None for o in data_objects)
        same_length = (len(
            set((o.number_values for o in data_objects))) == 1)
        if all_sized and same_length:
            return self._read_interleaved_sized(file, data_objects)
        else:
            return self._read_interleaved(file, data_objects)

    def _read_interleaved_sized(self, file, data_objects):
        """Read interleaved data where all channels have a sized data type and the same length
        """
        log.debug("Reading interleaved data all at once")

        total_data_width = sum(o.data_type.size for o in data_objects)
        log.debug("total_data_width: %d", total_data_width)

        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            file, total_data_width, data_objects[0].number_values)

        # Now get arrays for each channel
        channel_data = {}
        data_pos = 0
        for (i, obj) in enumerate(data_objects):
            byte_columns = tuple(
                range(data_pos, obj.data_type.size + data_pos))
            log.debug("Byte columns for channel %d: %s", i, byte_columns)
            # Select columns for this channel, so that number of values will
            # be number of bytes per point * number of data points.
            # Then use ravel to flatten the results into a vector.
            object_data = combined_data[:, byte_columns].ravel()
            if obj.data_type.nptype is not None:
                # Set correct data type, so that the array length should be correct
                object_data.dtype = (
                    obj.data_type.nptype.newbyteorder(self.endianness))
            else:
                object_data = obj.data_type.from_bytes(object_data, self.endianness)
            channel_data[obj.path] = object_data
            data_pos += obj.data_type.size

        return RawDataChunk.channel_data(channel_data)

    def _read_interleaved(self, file, data_objects):
        """Read interleaved data that doesn't have a numpy type"""

        log.debug("Reading interleaved data point by point")
        object_data = {}
        points_added = {}
        for obj in data_objects:
            object_data[obj.path] = obj.new_segment_data()
            points_added[obj.path] = 0
        while any([points_added[o.path] < o.number_values
                   for o in data_objects]):
            for obj in data_objects:
                if points_added[obj.path] < obj.number_values:
                    object_data[obj.path][points_added[obj.path]] = (
                        obj.read_value(file))
                    points_added[obj.path] += 1

        return RawDataChunk.channel_data(object_data)


class ContiguousDataSegment(BaseSegment):
    """ A TDMS segment with contiguous (non-interleaved) data
    """

    __slots__ = []

    def _new_segment_object(self, object_path):
        return TdmsSegmentObject(object_path, self.endianness)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        log.debug("Reading contiguous data chunk")
        object_data = {}
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            object_data[obj.path] = obj.read_values(file, number_values)
        return RawDataChunk.channel_data(object_data)

    def _read_channel_data_chunk(self, file, data_objects, chunk_index, channel_path):
        """ Read data from a chunk for a single channel
        """
        channel_data = RawChannelDataChunk.empty()
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            if obj.path == channel_path:
                channel_data = RawChannelDataChunk.channel_data(obj.read_values(file, number_values))
            elif number_values == obj.number_values:
                # Seek over data for other channel data
                file.seek(obj.data_size, os.SEEK_CUR)
            else:
                # In last chunk with reduced chunk size
                if obj.data_type.size is None:
                    # Type is unsized (eg. string), try reading number of values
                    obj.read_values(file, number_values)
                else:
                    file.seek(obj.data_type.size * number_values, os.SEEK_CUR)
        return channel_data

    def _get_channel_number_values(self, obj, chunk_index):
        if (chunk_index == (self.num_chunks - 1) and
                self.final_chunk_proportion != 1.0):
            return int(obj.number_values * self.final_chunk_proportion)
        else:
            return obj.number_values


class TdmsSegmentObject(BaseSegmentObject):
    """ A standard (non DAQmx) TDMS segment object
    """

    __slots__ = []

    def read_raw_data_index(self, f, raw_data_index_header):
        # Metadata format is standard (non-DAQmx) TDMS format.
        # raw_data_index_header gives the length of the index information.

        # Read the data type
        try:
            self.data_type = types.tds_data_types[
                types.Uint32.read(f, self.endianness)]
        except KeyError:
            raise KeyError("Unrecognised data type")
        log.debug("Object data type: %s", self.data_type.__name__)

        if (self.data_type.size is None and
                self.data_type != types.String):
            raise ValueError(
                "Unsupported data type: %r" % self.data_type)

        # Read data dimension
        dimension = types.Uint32.read(f, self.endianness)
        # In TDMS version 2.0, 1 is the only valid value for dimension
        if dimension != 1:
            raise ValueError("Data dimension is not 1")

        # Read number of values
        self.number_values = types.Uint64.read(f, self.endianness)

        # Variable length data types have total size
        if self.data_type in (types.String,):
            self.data_size = types.Uint64.read(f, self.endianness)
        else:
            self.data_size = self.number_values * self.data_type.size

        log.debug(
            "Object number of values in segment: %d", self.number_values)

    def read_value(self, file):
        """Read a single value from the given file"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=1)[0]
        return self.data_type.read(file, self.endianness)

    def read_values(self, file, number_values):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=number_values)
        elif self.data_type.size is not None:
            byte_data = fromfile(file, dtype=np.dtype('uint8'), count=number_values * self.data_type.size)
            return self.data_type.from_bytes(byte_data, self.endianness)
        else:
            return self.data_type.read_values(file, number_values, self.endianness)

    def new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values
