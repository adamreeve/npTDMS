""" Lower level TDMS reader API that allows finer grained reading of data
"""

from nptdms.utils import Timer, OrderedDict
from nptdms.tdms_segment import read_segment_metadata
from nptdms.log import log_manager

log = log_manager.get_logger(__name__)


class TdmsReader(object):
    """ Reads metadata and data from a TDMS file.

    :ivar object_metadata: Dictionary of object path to ObjectMetadata
    """

    def __init__(self, tdms_file):
        """ Initialise a new TdmsReader

        :param tdms_file: Either the path to the tdms file to read or an already
            opened file.
        """
        self._segments = None
        self._prev_segment_objects = {}
        self.object_metadata = OrderedDict()
        self._file_path = None

        if hasattr(tdms_file, "read"):
            # Is a file
            self._file = tdms_file
        else:
            # Is path to a file
            self._file = open(tdms_file, 'rb')
            self._file_path = tdms_file

    def close(self):
        if self._file_path is not None:
            # File path was provided so we opened the file and
            # should close it.
            self._file.close()
        # Otherwise always remove reference to the file
        self._file = None

    def read_metadata(self):
        """ Read all metadata and structure information from a TdmsFile
        """

        self._segments = []
        with Timer(log, "Read metadata"):
            # Read metadata first to work out how much space we need
            previous_segment = None
            while True:
                try:
                    segment = read_segment_metadata(
                        self._file, self._prev_segment_objects, previous_segment)
                except EOFError:
                    # We've finished reading the file
                    break

                self._update_object_metadata(segment)
                self._update_object_properties(segment)
                self._segments.append(segment)
                previous_segment = segment

                if segment.next_segment_pos is None:
                    break
                else:
                    self._file.seek(segment.next_segment_pos)

    def read_raw_data(self):
        """ Read raw data from all segments, chunk by chunk

        :returns: A generator that yields DataChunk objects
        """
        if self._segments is None:
            raise RuntimeError(
                "Cannot read data unless metadata has first been read")
        for segment in self._segments:
            for chunk in segment.read_raw_data(self._file):
                yield chunk

    def read_raw_data_for_channel(self, channel_path):
        """ Read raw data for a single channel, chunk by chunk

        :param channel_path: The path of the channel object to read data for
        :returns: A generator that yields ChannelDataChunk objects
        """
        if self._segments is None:
            raise RuntimeError(
                "Cannot read data unless metadata has first been read")
        for segment in self._segments:
            for chunk in segment.read_raw_data_for_channel(self._file, channel_path):
                yield chunk

    def _update_object_metadata(self, segment):
        """ Update object metadata using the metadata read from a single segment
        """
        for segment_object in segment.ordered_objects:
            path = segment_object.path
            self._prev_segment_objects[path] = segment_object

            object_metadata = self._get_or_create_object(path)
            if segment_object.has_data:
                object_metadata.num_values += _number_of_segment_values(segment_object, segment)
            _update_object_data_type(path, object_metadata, segment_object)
            _update_object_scaler_data_types(path, object_metadata, segment_object)

    def _update_object_properties(self, segment):
        """ Update object properties using any properties in a segment
        """
        if segment.object_properties is not None:
            for path, properties in segment.object_properties.items():
                object_metadata = self._get_or_create_object(path)
                for prop, val in properties:
                    object_metadata.properties[prop] = val

    def _get_or_create_object(self, path):
        """ Get existing object metadata or create metadata for a new object
        """
        try:
            return self.object_metadata[path]
        except KeyError:
            obj = ObjectMetadata()
            self.object_metadata[path] = obj
            return obj


def _number_of_segment_values(segment_object, segment):
    """ Compute the number of values an object has in a segment
    """
    num_chunks = segment.num_chunks
    final_chunk_proportion = segment.final_chunk_proportion
    if final_chunk_proportion == 1.0:
        return segment_object.number_values * num_chunks
    else:
        return (segment_object.number_values * (num_chunks - 1) +
                int(segment_object.number_values * final_chunk_proportion))


def _update_object_data_type(path, obj, segment_object):
    """ Update the data type for an object using its segment metadata
    """
    if obj.data_type is not None and obj.data_type != segment_object.data_type:
        raise ValueError(
            "Segment data doesn't have the same type as previous "
            "segments for objects %s. Expected type %s but got %s" %
            (path, obj.data_type, segment_object.data_type))
    obj.data_type = segment_object.data_type


def _update_object_scaler_data_types(path, obj, segment_object):
    """ Update the DAQmx scaler data types for an object using its segment metadata
    """
    if segment_object.scaler_data_types is not None:
        if obj.scaler_data_types is not None and obj.scaler_data_types != segment_object.scaler_data_types:
            raise ValueError(
                "Segment data doesn't have the same scaler data types as previous "
                "segments for objects %s. Expected types %s but got %s" %
                (path, obj.scaler_data_types, segment_object.scaler_data_types))
        obj.scaler_data_types = segment_object.scaler_data_types


class ObjectMetadata(object):
    """ Stores information about an object in a TDMS file
    """
    def __init__(self):
        self.properties = OrderedDict()
        self.data_type = None
        self.scaler_data_types = None
        self.num_values = 0
