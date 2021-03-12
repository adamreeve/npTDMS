""" Python module for reading TDMS files produced by LabView

    This module contains the public facing API for reading TDMS files
"""

from collections import defaultdict
import numpy as np

from nptdms import scaling, types
from nptdms.utils import Timer, OrderedDict, cached_property
from nptdms.log import log_manager
from nptdms.common import ObjectPath
from nptdms.reader import TdmsReader
from nptdms.channel_data import get_data_receiver, slice_raw_data
from nptdms.export import hdf_export, pandas_export
from nptdms.base_segment import RawChannelDataChunk
from nptdms.timestamp import TdmsTimestamp, TimestampArray


log = log_manager.get_logger(__name__)


class TdmsFile(object):
    """ Reads and stores data from a TDMS file.

    There are two main ways to create a new TdmsFile object.
    TdmsFile.read will read all data into memory::

        tdms_file = TdmsFile.read(tdms_file_path)

    or you can use TdmsFile.open to read file metadata but not immediately read all data,
    for cases where a file is too large to easily fit in memory or you don't need to
    read data for all channels::

        with TdmsFile.open(tdms_file_path) as tdms_file:
            # Use tdms_file
            ...

    This class acts like a dictionary, where the keys are names of groups in the TDMS
    files and the values are TdmsGroup objects.
    A TdmsFile can be indexed by group name to access a group within the TDMS file, for example::

        tdms_file = TdmsFile.read(tdms_file_path)
        group = tdms_file[group_name]

    Iterating over a TdmsFile produces the names of groups in this file,
    or you can use the groups method to directly access all groups::

        for group in tdms_file.groups():
            # Use group
            ...
    """

    @staticmethod
    def read(file, raw_timestamps=False, memmap_dir=None):
        """ Creates a new TdmsFile object and reads all data in the file

        :param file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param raw_timestamps: By default TDMS timestamps are read as numpy datetime64
            but this loses some precision.
            Setting this to true will read timestamps as a custom TdmsTimestamp type.
        :param memmap_dir: The directory to store memory mapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        """
        return TdmsFile(file, raw_timestamps=raw_timestamps, memmap_dir=memmap_dir)

    @staticmethod
    def open(file, raw_timestamps=False, memmap_dir=None):
        """ Creates a new TdmsFile object and reads metadata, leaving the file open
            to allow reading channel data

        :param file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param raw_timestamps: By default TDMS timestamps are read as numpy datetime64
            but this loses some precision.
            Setting this to true will read timestamps as a custom TdmsTimestamp type.
        :param memmap_dir: The directory to store memory mapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        """
        return TdmsFile(
            file, raw_timestamps=raw_timestamps, memmap_dir=memmap_dir, read_metadata_only=True, keep_open=True)

    @staticmethod
    def read_metadata(file, raw_timestamps=False):
        """ Creates a new TdmsFile object and only reads the metadata

        :param file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param raw_timestamps: By default TDMS timestamps are read as numpy datetime64
            but this loses some precision.
            Setting this to true will read timestamps as a custom TdmsTimestamp type.
        """
        return TdmsFile(file, raw_timestamps=raw_timestamps, read_metadata_only=True)

    def __init__(self, file, raw_timestamps=False, memmap_dir=None, read_metadata_only=False, keep_open=False):
        """Initialise a new TdmsFile object

        :param file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param raw_timestamps: By default TDMS timestamps are read as numpy datetime64
            but this loses some precision.
            Setting this to true will read timestamps as a custom TdmsTimestamp type.
        :param memmap_dir: The directory to store memory mapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        :param read_metadata_only: If this parameter is enabled then only the
            metadata of the TDMS file will read.
        :param keep_open: Keeps the file open so data can be read if only metadata
            is read initially.
        """

        self._memmap_dir = memmap_dir
        self._raw_timestamps = raw_timestamps
        self._groups = OrderedDict()
        self._properties = OrderedDict()
        self._channel_data = {}
        self.data_read = False

        self._reader = TdmsReader(file)
        try:
            self._read_file(self._reader, read_metadata_only, keep_open)
        finally:
            if not keep_open:
                self._reader.close()

    def groups(self):
        """Returns a list of the groups in this file

        :rtype: List of TdmsGroup.
        """

        return list(self._groups.values())

    @property
    def properties(self):
        """ Return the properties of this file as a dictionary

        These are the properties associated with the root TDMS object.
        """

        return self._properties

    def as_dataframe(self, time_index=False, absolute_time=False, scaled_data=True):
        """
        Converts the TDMS file to a DataFrame. DataFrame columns are named using the TDMS object paths.

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :param scaled_data: By default the scaled data will be used.
            Set to False to use raw unscaled data.
            For DAQmx data, there will be one column per DAQmx raw scaler and column names will include the scale id.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """

        return pandas_export.from_tdms_file(self, time_index, absolute_time, scaled_data)

    def as_hdf(self, filepath, mode='w', group='/'):
        """
        Converts the TDMS file into an HDF5 file

        :param filepath: The path of the HDF5 file you want to write to.
        :param mode: The write mode of the HDF5 file. This can be 'w' or 'a'
        :param group: A group in the HDF5 file that will contain the TDMS data.
        """
        return hdf_export.from_tdms_file(self, filepath, mode, group)

    def data_chunks(self):
        """ A generator that streams chunks of data from disk.
        This method may only be used when the TDMS file was opened without reading all data immediately.

        :rtype: Generator that yields :class:`DataChunk` objects
        """
        channel_offsets = defaultdict(int)
        for chunk in self._reader.read_raw_data():
            _convert_data_chunk(chunk, self._raw_timestamps)
            yield DataChunk(self, chunk, channel_offsets)
            for path, data in chunk.channel_data.items():
                channel_offsets[path] += len(data)

    def close(self):
        """ Close the underlying file if it was opened by this TdmsFile

            If this TdmsFile was initialised with an already open file
            then the reference to it is released but the file is not closed.
        """
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __len__(self):
        """ Returns the number of groups in this file
        """
        return len(self._groups)

    def __iter__(self):
        """ Returns an iterator over the names of groups in this file
        """
        return iter(self._groups)

    def __getitem__(self, group_name):
        """ Retrieve a TDMS group from the file by name
        """
        try:
            return self._groups[group_name]
        except KeyError:
            raise KeyError("There is no group named '%s' in the TDMS file" % group_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _read_file(self, tdms_reader, read_metadata_only, keep_open):
        tdms_reader.read_metadata(require_segment_indexes=keep_open)

        # Use object metadata to build group and channel objects
        group_properties = OrderedDict()
        group_channels = OrderedDict()
        object_properties = {
            path_string: self._convert_properties(obj.properties)
            for path_string, obj in tdms_reader.object_metadata.items()}
        try:
            self._properties = object_properties['/']
        except KeyError:
            pass

        for (path_string, obj) in tdms_reader.object_metadata.items():
            properties = object_properties[path_string]
            path = ObjectPath.from_string(path_string)
            if path.is_root:
                pass
            elif path.is_group:
                group_properties[path.group] = properties
            else:
                # Object is a channel
                try:
                    channel_group_properties = object_properties[path.group_path()]
                except KeyError:
                    channel_group_properties = OrderedDict()
                channel = TdmsChannel(
                    path, obj.data_type, obj.scaler_data_types, obj.num_values,
                    properties, channel_group_properties, self._properties,
                    tdms_reader, self._raw_timestamps, self._memmap_dir)
                if path.group in group_channels:
                    group_channels[path.group].append(channel)
                else:
                    group_channels[path.group] = [channel]

        # Create group objects containing channels and properties
        for group_name, properties in group_properties.items():
            try:
                channels = group_channels[group_name]
            except KeyError:
                channels = []
            group_path = ObjectPath(group_name)
            self._groups[group_name] = TdmsGroup(group_path, properties, channels)
        for group_name, channels in group_channels.items():
            if group_name not in self._groups:
                # Group with channels but without any corresponding object metadata in the file:
                group_path = ObjectPath(group_name)
                self._groups[group_name] = TdmsGroup(group_path, {}, channels)

        if not read_metadata_only:
            self._read_data(tdms_reader)

    def _read_data(self, tdms_reader):
        with Timer(log, "Allocate space"):
            # Allocate space for data
            for group in self.groups():
                for channel in group.channels():
                    self._channel_data[channel.path] = get_data_receiver(
                        channel, len(channel), self._raw_timestamps, self._memmap_dir)

        with Timer(log, "Read data"):
            # Now actually read all the data
            for chunk in tdms_reader.read_raw_data():
                for (path, data) in chunk.channel_data.items():
                    channel_data = self._channel_data[path]
                    if data.data is not None:
                        channel_data.append_data(data.data)
                    elif data.scaler_data is not None:
                        for scaler_id, scaler_data in data.scaler_data.items():
                            channel_data.append_scaler_data(scaler_id, scaler_data)

            for group in self.groups():
                for channel in group.channels():
                    channel_data = self._channel_data[channel.path]
                    if channel_data is not None:
                        channel._set_raw_data(channel_data)

        self.data_read = True

    def _convert_properties(self, properties):
        def convert_prop(val):
            if isinstance(val, TdmsTimestamp) and not self._raw_timestamps:
                # Convert timestamps to numpy datetime64 if raw timestamps are not requested
                return val.as_datetime64()
            return val
        return OrderedDict((k, convert_prop(v)) for (k, v) in properties.items())


class TdmsGroup(object):
    """ Represents a group of channels in a TDMS file.

    This class acts like a dictionary, where the keys are names of channels in the group
    and the values are TdmsChannel objects.
    A TdmsGroup can be indexed by channel name to access a channel in this group, for example::

        channel = group[channel_name]

    Iterating over a TdmsGroup produces the names of channels in this group,
    or you can use the channels method to directly access all channels::

        for channel in group.channels():
            # Use channel
            ...

    :ivar ~.properties: Dictionary of TDMS properties defined for this group.
    """

    def __init__(self, path, properties, channels):
        self._path = path
        self.properties = properties
        self._channels = {c.name: c for c in channels}

    def __repr__(self):
        return "<TdmsGroup with path %s>" % self.path

    @property
    def path(self):
        """ Path to the TDMS object for this group
        """
        return str(self._path)

    @property
    def name(self):
        """ The name of this group
        """
        return self._path.group

    def channels(self):
        """ The list of channels in this group

        :rtype: A list of TdmsChannel
        """
        return list(self._channels.values())

    def as_dataframe(self, time_index=False, absolute_time=False, scaled_data=True):
        """
        Converts the TDMS group to a DataFrame. DataFrame columns are named using the channel names.

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :param scaled_data: By default the scaled data will be used.
            Set to False to use raw unscaled data.
            For DAQmx data, there will be one column per DAQmx raw scaler and column names will include the scale id.
        :return: The TDMS object data.
        :rtype: pandas.DataFrame
        """

        return pandas_export.from_group(self, time_index, absolute_time, scaled_data)

    def __len__(self):
        """ Returns the number of channels in this group
        """
        return len(self._channels)

    def __iter__(self):
        """ Returns an iterator over the names of channels in this group
        """
        return iter(self._channels)

    def __getitem__(self, channel_name):
        """ Retrieve a TDMS channel from this group by name
        """
        try:
            return self._channels[channel_name]
        except KeyError:
            raise KeyError(
                "There is no channel named '%s' in group '%s' of the TDMS file" %
                (channel_name, self.name))


class TdmsChannel(object):
    """ Represents a data channel in a TDMS file.

    This class acts like an array, you can get the length of a channel using :code:`len(channel)`,
    and can iterate over values in the channel using a for loop,
    or index into a channel using an integer index to get a single value::

        for value in channel:
            # Use value
            ...
        first_value = channel[0]

    Or you can index using a slice to retrieve a range of data as a numpy array.
    To get all data in this channel as a numpy array::

        all_data = channel[:]

    Or to retrieve a subset of data::

        data_subset = channel[start:stop]

    :ivar ~.properties: Dictionary of TDMS properties defined for this channel,
                      for example the start time and time increment for waveforms.
    """

    def __init__(
            self, path, data_type, scaler_data_types, number_values,
            properties, group_properties, file_properties,
            tdms_reader, raw_timestamps, memmap_dir):
        self._path = path
        self.properties = properties
        self._length = number_values
        self.data_type = data_type
        self.scaler_data_types = scaler_data_types
        self._group_properties = group_properties
        self._file_properties = file_properties
        self._reader = tdms_reader
        self._raw_timestamps = raw_timestamps
        self._memmap_dir = memmap_dir

        self._raw_data = None
        self._cached_chunk = None
        self._cached_chunk_bounds = None

    def __repr__(self):
        return "<TdmsChannel with path %s>" % self.path

    def __len__(self):
        """ Returns the number of values in this channel
        """
        return self._length

    def __iter__(self):
        """ Returns an iterator over the values in this channel
        """
        if self._raw_data is not None:
            return iter(self.data)
        else:
            return self._read_data_values()

    def __getitem__(self, index):
        if self._raw_data is not None:
            return self.data[index]
        elif index is Ellipsis:
            return self.read_data()
        elif isinstance(index, slice):
            return self._read_slice(index.start, index.stop, index.step)
        elif isinstance(index, int):
            return self._read_at_index(index)
        else:
            raise TypeError("Invalid index type '%s', expected int, slice or Ellipsis" % type(index).__name__)

    @property
    def path(self):
        """ Path to the TDMS object for this channel
        """
        return str(self._path)

    @property
    def name(self):
        """ The name of this channel
        """
        return self._path.channel

    @property
    def group_name(self):
        """ The name of the group that contains this channel
        """
        return self._path.group

    @cached_property
    def dtype(self):
        """ NumPy data type of the channel data

        For data with a scaling this is the data type of the scaled data

        :rtype: numpy.dtype
        """
        channel_scaling = self._scaling
        if channel_scaling is not None:
            return channel_scaling.get_dtype(self.data_type, self.scaler_data_types)
        return self._raw_data_dtype()

    def _raw_data_dtype(self):
        if self.data_type is types.String:
            return np.dtype('O')
        elif self.data_type is types.TimeStamp:
            return np.dtype('<M8[us]')
        if self.data_type is not None and self.data_type.nptype is not None:
            return self.data_type.nptype
        return np.dtype('V8')

    @cached_property
    def data(self):
        """ If the TdmsFile was created by reading all data, this property
        provides direct access to the numpy array containing the data for this channel.

        Indexing into the channel with a slice should be preferred to using this property, for example::

            channel_data = channel[:]
        """
        if len(self) > 0 and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        if self._raw_data is None:
            return np.empty((0, ), dtype=self.dtype)
        return self._scale_data(self._raw_data)

    @property
    def raw_data(self):
        """ If the TdmsFile was created by reading all data, this property
        provides direct access to the numpy array of raw, unscaled data.
        For unscaled objects this is the same as the data property.
        """
        if len(self) > 0 and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        if self._raw_data is None:
            return np.empty((0, ), dtype=self._raw_data_dtype())
        if self._raw_data.scaler_data:
            if len(self._raw_data.scaler_data) == 1:
                return next(v for v in self._raw_data.scaler_data.values())
            else:
                raise Exception(
                    "This object has data for multiple DAQmx scalers, "
                    "use the raw_scaler_data property to get raw data "
                    "for a scale_id")
        return self._raw_data.data

    @property
    def raw_scaler_data(self):
        """ If the TdmsFile was created by reading all data, this property
        provides direct access to the numpy array of raw DAQmx scaler data
        as a dictionary mapping from scale id to raw data arrays.
        """
        if len(self) > 0 and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        return self._raw_data.scaler_data

    def data_chunks(self):
        """ A generator that streams chunks data for this channel from disk.
        This method may only be used when the TDMS file was opened without reading all data immediately.

        :rtype: Generator that yields :class:`ChannelDataChunk` objects
        """
        channel_offset = 0
        for raw_data_chunk in self._read_channel_data_chunks():
            yield ChannelDataChunk(self, raw_data_chunk, channel_offset)
            channel_offset += len(raw_data_chunk)

    def read_data(self, offset=0, length=None, scaled=True):
        """ Reads data for this channel from the TDMS file and returns it as a numpy array

        Indexing into the channel with a slice should be preferred over using
        this method, but this method is needed if you want to read raw, unscaled data.

        :param offset: Initial position to read data from.
        :param length: Number of values to attempt to read.
            Fewer values will be returned if attempting to read beyond the end of the available data.
        :param scaled: By default scaling will be applied to the returned data.
            Set this parameter to False to return raw unscaled data.
            For DAQmx data a dictionary of scaler id to raw scaler data will be returned.
        """
        if self._raw_data is None:
            raw_data = self._read_channel_data(offset, length)
        else:
            raw_data = slice_raw_data(self._raw_data, offset, length)

        if raw_data is None:
            dtype = self.dtype if scaled else self._raw_data_dtype()
            return np.empty((0,), dtype=dtype)
        if scaled:
            return self._scale_data(raw_data)
        else:
            if raw_data.scaler_data:
                return raw_data.scaler_data
            return raw_data.data

    def time_track(self, absolute_time=False, accuracy='ns'):
        """Return an array of time or the independent variable for this channel

        This depends on the object having the wf_increment
        and wf_start_offset properties defined.
        Note that wf_start_offset is usually zero for time-series data.
        If you have time-series data channels with different start times,
        you should use the absolute time or calculate the time offsets using
        the wf_start_time property.

        For larger timespans, the accuracy setting should be set lower.
        The default setting is 'ns', which has a timespan of
        [1678 AD, 2262 AD]. For the exact ranges, refer to
        http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
        section "Datetime Units".

        :param absolute_time: Whether the returned time values are absolute
            times rather than relative to the start time. If true, the
            wf_start_time property must be set.
        :param accuracy: The accuracy of the returned datetime64 array.
        :rtype: NumPy array.
        :raises: KeyError if required properties aren't found

        """

        try:
            increment = self.properties['wf_increment']
            offset = self.properties['wf_start_offset']
        except KeyError:
            raise KeyError("Object does not have time properties available.")

        relative_time = np.linspace(
            offset,
            offset + (len(self) - 1) * increment,
            len(self))

        if not absolute_time:
            return relative_time

        try:
            start_time = self.properties['wf_start_time']
        except KeyError:
            raise KeyError(
                "Object does not have start time property available.")

        try:
            unit_correction = {
                's': 1e0,
                'ms': 1e3,
                'us': 1e6,
                'ns': 1e9,
            }[accuracy]
        except KeyError:
            raise KeyError("Invalid accuracy: {0}".format(accuracy))

        # Because numpy only knows ints as its date datatype,
        # convert to accuracy.
        time_type = "timedelta64[{0}]".format(accuracy)
        return (np.datetime64(start_time) +
                (relative_time * unit_correction).astype(time_type))

    def as_dataframe(self, time_index=False, absolute_time=False, scaled_data=True):
        """
        Converts the TDMS channel to a DataFrame. The DataFrame column is named using the channel path.

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :param scaled_data: By default the scaled data will be used.
            Set to False to use raw unscaled data.
            For DAQmx data, there will be one column per DAQmx raw scaler and column names will include the scale id.
        :return: The TDMS object data.
        :rtype: pandas.DataFrame
        """

        return pandas_export.from_channel(self, time_index, absolute_time, scaled_data)

    def _read_data_values(self):
        for chunk in self.data_chunks():
            for value in chunk:
                yield value

    def _read_slice(self, start, stop, step):
        if step == 0:
            raise ValueError("Step size cannot be zero")

        # Replace None values with defaults
        step = 1 if step is None else step
        if start is None:
            start = 0 if step > 0 else -1
        if stop is None:
            stop = self._length if step > 0 else -1 - self._length

        # Handle negative indices
        if start < 0:
            start = self._length + start
        if stop < 0:
            stop = self._length + stop

        # Check for empty ranges
        if stop == start:
            return np.empty((0, ), dtype=self.dtype)
        if step > 0 and (stop < start or start >= self._length or stop < 0):
            return np.empty((0,), dtype=self.dtype)
        if step < 0 and (stop > start or stop >= self._length or start < 0):
            return np.empty((0,), dtype=self.dtype)

        # Trim values outside bounds
        if start < 0:
            start = 0
        if start >= self._length:
            start = self._length - 1
        if stop > self._length:
            stop = self._length
        if stop < -1:
            stop = -1

        # Read data and handle step size
        if step > 0:
            read_data = self.read_data(start, stop - start)
            return read_data[::step] if step > 1 else read_data
        else:
            read_data = self.read_data(stop + 1, start - stop)
            return read_data[::step]

    def _read_at_index(self, index):
        orig_index = index
        if index < 0:
            index = self._length + index
        if index < 0 or index >= self._length:
            raise IndexError("Index {0} is outside of the channel bounds [0, {1}]".format(orig_index, self._length - 1))

        if self._cached_chunk is not None:
            # Check if we've already read and cached the chunk containing this index
            bounds = self._cached_chunk_bounds
            if bounds[0] <= index < bounds[1]:
                return self._cached_chunk[index - bounds[0]]

        chunk, chunk_offset = self._read_channel_data_chunk_for_index(index)
        scaled_chunk = self._scale_data(chunk)
        self._cached_chunk = scaled_chunk
        self._cached_chunk_bounds = (chunk_offset, chunk_offset + len(scaled_chunk))

        return scaled_chunk[index - chunk_offset]

    def _scale_data(self, raw_data):
        scale = self._scaling
        if scale is not None:
            return scale.scale(raw_data)
        elif raw_data.scaler_data:
            raise ValueError("Missing scaling information for DAQmx data")
        else:
            return raw_data.data

    @cached_property
    def _scaling(self):
        return scaling.get_scaling(
            self.properties, self._group_properties, self._file_properties)

    def _read_channel_data_chunks(self):
        for chunk in self._reader.read_raw_data_for_channel(self.path):
            _convert_channel_data_chunk(chunk, self._raw_timestamps)
            yield chunk

    def _read_channel_data_chunk_for_index(self, index):
        (chunk, offset) = self._reader.read_channel_chunk_for_index(self.path, index)
        _convert_channel_data_chunk(chunk, self._raw_timestamps)
        return chunk, offset

    def _read_channel_data(self, offset=0, length=None):
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if length is not None and length < 0:
            raise ValueError("length must be non-negative")

        with Timer(log, "Allocate space for channel"):
            # Allocate space for data
            if length is None:
                num_values = len(self) - offset
            else:
                num_values = min(length, len(self) - offset)
            num_values = max(0, num_values)
            channel_data = get_data_receiver(self, num_values, self._raw_timestamps, self._memmap_dir)

        with Timer(log, "Read data for channel"):
            # Now actually read all the data
            for chunk in self._reader.read_raw_data_for_channel(self.path, offset, length):
                if chunk.data is not None:
                    channel_data.append_data(chunk.data)
                if chunk.scaler_data is not None:
                    for scaler_id, scaler_data in chunk.scaler_data.items():
                        channel_data.append_scaler_data(scaler_id, scaler_data)

        return channel_data

    def _set_raw_data(self, data):
        self._raw_data = data


class DataChunk(object):
    """ A chunk of data in a TDMS file

    Can be indexed by group name to get the data for a group in this channel,
    which can then be indexed by channel name to get the data for a channel in this chunk.
    For example::

        group_chunk = data_chunk[group_name]
        channel_chunk = group_chunk[channel_name]
    """
    def __init__(self, tdms_file, raw_data_chunk, channel_offsets):
        self._groups = OrderedDict(
            (group.name, GroupDataChunk(tdms_file, group, raw_data_chunk, channel_offsets))
            for group in tdms_file.groups())

    def __getitem__(self, group_name):
        """ Get a chunk of data for a group
        """
        return self._groups[group_name]

    def groups(self):
        """ Returns chunks of data for all groups

        :rtype: List of :class:`GroupDataChunk`
        """
        return list(self._groups.values())


class GroupDataChunk(object):
    """ A chunk of data for a group in a TDMS file

    Can be indexed by channel name to get the data for a channel in this chunk.
    For example::

        channel_chunk = group_chunk[channel_name]

    :ivar ~.name: Name of the group
    """
    def __init__(self, tdms_file, group, raw_data_chunk, channel_offsets):
        self.name = group.name
        self._channels = OrderedDict(
            (channel.name, ChannelDataChunk(
                channel,
                raw_data_chunk.channel_data.get(channel.path, RawChannelDataChunk.empty()),
                channel_offsets[channel.path]))
            for channel in group.channels())

    def __getitem__(self, channel_name):
        """ Get a chunk of data for a channel in this group
        """
        return self._channels[channel_name]

    def channels(self):
        """ Returns chunks of channel data for all channels in this group

        :rtype: List of :class:`ChannelDataChunk`
        """
        return list(self._channels.values())


class ChannelDataChunk(object):
    """ A chunk of data for a channel in a TDMS file

    Is an array-like object that supports indexing to access data, for example::

        chunk_length = len(channel_data_chunk)
        chunk_data = channel_data_chunk[:]

    :ivar ~.name: Name of the channel
    :ivar ~.offset: Starting index of this chunk of data in the entire channel
    """
    def __init__(self, channel, raw_data_chunk, offset):
        self._path = channel._path
        self._channel = channel
        self.name = channel.name
        self.offset = offset
        self._raw_data = raw_data_chunk

    def __len__(self):
        """ Returns the number of values in this chunk
        """
        return len(self._raw_data)

    def __getitem__(self, index):
        """ Get a value or slice of values from this chunk
        """
        return self._data()[index]

    def __iter__(self):
        """ Iterate over values in this chunk
        """
        return iter(self._data())

    def _data(self):
        if self._raw_data.data is None and self._raw_data.scaler_data is None:
            return np.empty((0, ), dtype=self._channel.dtype)

        scale = self._channel._scaling
        if scale is not None:
            return scale.scale(self._raw_data)
        elif self._raw_data.scaler_data:
            raise ValueError("Missing scaling information for DAQmx data")
        else:
            return self._raw_data.data


def _convert_data_chunk(chunk, raw_timestamps):
    for channel_chunk in chunk.channel_data.values():
        _convert_channel_data_chunk(channel_chunk, raw_timestamps)


def _convert_channel_data_chunk(channel_chunk, raw_timestamps):
    if not raw_timestamps and isinstance(channel_chunk.data, TimestampArray):
        channel_chunk.data = channel_chunk.data.as_datetime64()
