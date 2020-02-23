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

        :param tdms_file: An opened file object.
        """
        self._file = tdms_file
        self._segments = None
        self._prev_segment_objects = {}
        self.object_metadata = OrderedDict()

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

    def _update_object_metadata(self, segment):
        num_chunks = segment.num_chunks
        final_chunk_proportion = segment.final_chunk_proportion
        for segment_object in segment.ordered_objects:
            path = segment_object.path
            self._prev_segment_objects[path] = segment_object

            try:
                obj = self.object_metadata[path]
            except KeyError:
                obj = ObjectMetadata()
                self.object_metadata[path] = obj
            for prop, val in segment_object.properties.items():
                obj.properties[prop] = val
            if segment_object.has_data:
                if final_chunk_proportion == 1.0:
                    obj.num_values += segment_object.number_values * num_chunks
                else:
                    obj.num_values += (
                        segment_object.number_values * (num_chunks - 1) +
                        int(segment_object.number_values *
                            final_chunk_proportion))
            if (obj.data_type is not None and
                    obj.data_type != segment_object.data_type):
                raise ValueError(
                    "Segment data doesn't have the same type as previous "
                    "segments for objects %s. Expected type %s but got %s" %
                    (path, obj.data_type, segment_object.data_type))
            obj.data_type = segment_object.data_type
            if segment_object.scaler_data_types is not None:
                if (obj.scaler_data_types is not None and
                        obj.scaler_data_types != segment_object.scaler_data_types):
                    raise ValueError(
                        "Segment data doesn't have the same scaler data types as previous "
                        "segments for objects %s. Expected types %s but got %s" %
                        (path, obj.scaler_data_types, segment_object.scaler_data_types))
                obj.scaler_data_types = segment_object.scaler_data_types


class ObjectMetadata(object):
    def __init__(self):
        self.properties = {}
        self.data_type = None
        self.scaler_data_types = None
        self.num_values = 0
