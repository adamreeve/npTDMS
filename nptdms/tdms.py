""" Python module for reading TDMS files produced by LabView

    This module contains the public facing API
"""

import numpy as np

from nptdms import scaling, types
from nptdms.utils import Timer, OrderedDict
from nptdms.log import log_manager
from nptdms.common import path_components
from nptdms.reader import TdmsReader
from nptdms.channel_data import get_data_receiver


log = log_manager.get_logger(__name__)


# Have to get a reference to the builtin property decorator
# so we can use it in TdmsObject, which has a property method.
_property_builtin = property


class TdmsFile(object):
    """Reads and stores data from a TDMS file.
    """

    @staticmethod
    def read(file, memmap_dir=None):
        """ Creates a new TdmsFile object and reads all data in the file

        :param file: Either the path to the tdms file to read or an already
            opened file.
        :param memmap_dir: The directory to store memory mapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        """
        return TdmsFile(file, memmap_dir=memmap_dir)

    @staticmethod
    def open(file, memmap_dir=None):
        """ Creates a new TdmsFile object and reads metadata, leaving the file open
            to allow reading data

        :param file: Either the path to the tdms file to read or an already
            opened file.
        :param memmap_dir: The directory to store memory mapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        """
        return TdmsFile(file, memmap_dir=memmap_dir, read_metadata_only=True, keep_open=True)

    @staticmethod
    def read_metadata(file):
        """ Creates a new TdmsFile object and only reads the metadata

        :param file: Either the path to the tdms file to read or an already
            opened file.
        """
        return TdmsFile(file, read_metadata_only=True)

    def __init__(self, file, memmap_dir=None, read_metadata_only=False, keep_open=False):
        """Initialise a new TDMS file object

        :param file: Either the path to the tdms file to read or an already
            opened file.
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
        self._channel_data = {}
        self._objects = OrderedDict()
        self._reader = None

        reader = TdmsReader(file)
        try:
            self._read_file(reader, read_metadata_only)
        finally:
            if keep_open:
                self._reader = reader
            else:
                reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """ Close the underlying file if it was opened by this TdmsFile

            If this TdmsFile was initialised with an already open file
            then the reference to it is released but the file is not closed.
        """
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _read_file(self, tdms_reader, read_metadata_only):
        tdms_reader.read_metadata()

        for (path, obj) in tdms_reader.object_metadata.items():
            self._objects[path] = TdmsObject(
                self, path, obj.properties, obj.data_type,
                obj.scaler_data_types, obj.num_values)

        if not read_metadata_only:
            self._read_data(tdms_reader)

    def _read_data(self, tdms_reader):
        with Timer(log, "Allocate space"):
            # Allocate space for data
            for (path, obj) in self._objects.items():
                self._channel_data[path] = get_data_receiver(obj, obj.number_values, self._memmap_dir)

        with Timer(log, "Read data"):
            # Now actually read all the data
            for chunk in tdms_reader.read_raw_data():
                for (path, data) in chunk.raw_data.items():
                    channel_data = self._channel_data[path]
                    channel_data.append_data(data)
                for (path, data) in chunk.daqmx_raw_data.items():
                    channel_data = self._channel_data[path]
                    for scaler_id, scaler_data in data.items():
                        channel_data.append_scaler_data(
                            scaler_id, scaler_data)

            for (path, obj) in self._objects.items():
                channel_data = self._channel_data[path]
                if channel_data is not None:
                    obj._set_raw_data(channel_data)

    def _read_channel_data(self, channel_path, offset=0, length=None):
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if length is not None and length < 0:
            raise ValueError("length must be non-negative")
        if self._reader is None:
            raise RuntimeError(
                "Cannot read channel data after the underlying TDMS reader is closed")

        obj = self._objects[channel_path]
        with Timer(log, "Allocate space"):
            # Allocate space for data
            if length is None:
                num_values = obj.number_values - offset
            else:
                num_values = min(length, obj.number_values - offset)
            num_values = max(0, num_values)
            channel_data = get_data_receiver(obj, num_values, self._memmap_dir)

        with Timer(log, "Read data"):
            # Now actually read all the data
            for chunk in self._reader.read_raw_data_for_channel(channel_path, offset, length):
                if chunk.raw_data is not None:
                    channel_data.append_data(chunk.raw_data)
                if chunk.daqmx_raw_data is not None:
                    for scaler_id, scaler_data in chunk.daqmx_raw_data.items():
                        channel_data.append_scaler_data(scaler_id, scaler_data)

        return channel_data

    def object(self, *path):
        """Get a TDMS object from the file

        :param path: The object group and channel. Providing no channel
            returns a group object, and providing no channel or group
            will return the root object.
        :rtype: :class:`TdmsObject`

        For example, to get the root object::

            object()

        To get a group::

            object("group_name")

        To get a channel::

            object("group_name", "channel_name")
        """

        object_path = _components_to_path(*path)
        try:
            return self._objects[object_path]
        except KeyError:
            raise KeyError("Invalid object path: %s" % object_path)

    def groups(self):
        """Return the names of groups in the file

        Note that there is not necessarily a TDMS object associated with
        each group name.

        :rtype: List of strings.

        """

        # Split paths into components and take the first (group) component.
        object_paths = (
            path_components(path)
            for path in self._objects)
        group_names = (path[0] for path in object_paths if len(path) > 0)

        # Use an ordered dict as an ordered set to find unique
        # groups in order.
        groups_set = OrderedDict()
        for group in group_names:
            groups_set[group] = None
        return list(groups_set)

    def group_channels(self, group):
        """Returns a list of channel objects for the given group

        :param group: Name of the group to get channels for.
        :rtype: List of :class:`TdmsObject` objects.

        """

        path = _components_to_path(group)
        return [
            self._objects[p]
            for p in self._objects
            if p.startswith(path + '/')]

    def channel_data(self, group, channel):
        """Get the data for a channel

        :param group: The name of the group the channel is in.
        :param channel: The name of the channel to get data for.
        :returns: The channel data.
        :rtype: NumPy array.

        """

        if self._reader is None:
            # Data should have already been loaded
            return self.object(group, channel).data
        else:
            # Data must be lazily loaded
            return self.object(group, channel).read_data()

    @_property_builtin
    def properties(self):
        """ Return the properties of this file as a dictionary

        These are the properties associated with the root TDMS object.
        """

        try:
            obj = self.object()
            return obj.properties
        except KeyError:
            return {}

    @_property_builtin
    def objects(self):
        """ A dictionary of objects in the TDMS file, where the keys are the object paths.
        """
        return self._objects

    def as_dataframe(self, time_index=False, absolute_time=False):
        """
        Converts the TDMS file to a DataFrame

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        dataframe_dict = OrderedDict()
        for key, value in self._objects.items():
            if value.has_data:
                index = value.time_track(absolute_time) if time_index else None
                dataframe_dict[key] = pd.Series(data=value.data, index=index)
        return pd.DataFrame.from_dict(dataframe_dict)

    def as_hdf(self, filepath, mode='w', group='/'):
        """
        Converts the TDMS file into an HDF5 file

        :param filepath: The path of the HDF5 file you want to write to.
        :param mode: The write mode of the HDF5 file. This can be w, a ...
        :param group: A group in the HDF5 file that will contain the TDMS data.
        """
        import h5py

        # Groups in TDMS are mapped to the first level of the HDF5 hierarchy

        # Channels in TDMS are then mapped to the second level of the HDF5
        # hierarchy, under the appropriate groups.

        # Properties in TDMS are mapped to attributes in HDF5.
        # These all exist under the appropriate, channel group etc.

        h5file = h5py.File(filepath, mode)

        container_group = None
        if group in h5file:
            container_group = h5file[group]
        else:
            container_group = h5file.create_group(group)

        # First write the properties at the root level
        try:
            root = self.object()
            for property_name, property_value in root.properties.items():
                container_group.attrs[property_name] = _hdf_attr_value(property_value)
        except KeyError:
            # No root object present
            pass

        # Now iterate through groups and channels,
        # writing the properties and data
        for group_name in self.groups():
            try:
                group = self.object(group_name)

                # Write the group's properties
                container_group.create_group(group_name)
                for prop_name, prop_value in group.properties.items():
                    container_group[group_name].attrs[prop_name] = _hdf_attr_value(prop_value)

            except KeyError:
                # No group object present
                pass

            # Write properties and data for each channel
            for channel in self.group_channels(group_name):
                channel_key = group_name + '/' + channel.channel

                if channel.data_type is types.String:
                    # Encode as variable length UTF-8 strings
                    channel_data = container_group.create_dataset(
                        channel_key, (len(channel.data), ), dtype=h5py.string_dtype())
                    channel_data[...] = channel.data
                elif channel.data_type is types.TimeStamp:
                    # Timestamps are represented as fixed length ASCII strings
                    # because HDF doesn't natively support timestamps
                    channel_data = container_group.create_dataset(
                        channel_key, (len(channel.data), ), dtype='S27')
                    string_data = np.datetime_as_string(channel.data, unit='us', timezone='UTC')
                    encoded_data = [s.encode('ascii') for s in string_data]
                    channel_data[...] = encoded_data
                else:
                    container_group[channel_key] = channel.data

                for prop_name, prop_value in channel.properties.items():
                    container_group[channel_key].attrs[prop_name] = _hdf_attr_value(prop_value)

        return h5file


class TdmsObject(object):
    """Represents an object in a TDMS file.

    :ivar path: The TDMS object path.
    :ivar properties: Dictionary of TDMS properties defined for this object,
                      for example the start time and time increment for
                      waveforms.
    :ivar has_data: Boolean, true if there is data associated with the object.

    """

    def __init__(
            self, tdms_file, path, properties, data_type=None,
            scaler_data_types=None, number_values=0):
        self.tdms_file = tdms_file
        self.path = path
        self.properties = properties
        self.number_values = number_values
        self.data_type = data_type
        self.scaler_data_types = scaler_data_types
        self.has_data = number_values > 0

        self._raw_data = None
        self._data_scaled = None

    def __repr__(self):
        return "<TdmsObject with path %s>" % self.path

    def property(self, property_name):
        """Returns the value of a TDMS property

        :param property_name: The name of the property to get.
        :returns: The value of the requested property.
        :raises: KeyError if the property isn't found.

        """

        try:
            return self.properties[property_name]
        except KeyError:
            raise KeyError(
                "Object does not have property '%s'" % property_name)

    @_property_builtin
    def group(self):
        """ Returns the name of the group for this object,
            or None if it is the root object.
        """
        path = path_components(self.path)
        if len(path) > 0:
            return path[0]
        return None

    @_property_builtin
    def channel(self):
        """ Returns the name of the channel for this object,
            or None if it is a group or the root object.
        """
        path = path_components(self.path)
        if len(path) > 1:
            return path[1]
        return None

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
            increment = self.property('wf_increment')
            offset = self.property('wf_start_offset')
        except KeyError:
            raise KeyError("Object does not have time properties available.")

        relative_time = np.linspace(
            offset,
            offset + (self.number_values - 1) * increment,
            self.number_values)

        if not absolute_time:
            return relative_time

        try:
            start_time = self.property('wf_start_time')
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

    def as_dataframe(self, absolute_time=False, scaled_data=True):
        """
        Converts the TDMS object to a DataFrame

        :param absolute_time: Whether times should be absolute rather than
            relative to the start time.
        :param scaled_data: Whether to return scaled or raw data.
        :return: The TDMS object data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        def get_data(chan):
            if scaled_data:
                return chan.data
            else:
                return chan.raw_data

        # When absolute_time is True,
        # use the wf_start_time as offset for the time_track()
        try:
            time = self.time_track(absolute_time)
        except KeyError:
            time = None
        if self.channel is None:
            return pd.DataFrame.from_dict(OrderedDict(
                (ch.channel, pd.Series(get_data(ch)))
                for ch in self.tdms_file.group_channels(self.group)))
        else:
            return pd.DataFrame(
                get_data(self), index=time, columns=[self.path])

    @_property_builtin
    def data(self):
        """
        NumPy array containing data if there is data for this object,
        otherwise None.
        """
        if self.has_data and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        if self._raw_data is None:
            return np.empty((0, 1))
        if self._data_scaled is None:
            self._data_scaled = self._scale_data(self._raw_data)
        return self._data_scaled

    @_property_builtin
    def raw_data(self):
        """
        The raw, unscaled data array.
        For unscaled objects this is the same as the data property.
        """
        if self.has_data and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        if self._raw_data is None:
            return np.empty((0, 1))
        if self._raw_data.scaler_data:
            if len(self._raw_data.scaler_data) == 1:
                return next(v for v in self._raw_data.scaler_data.values())
            else:
                raise Exception(
                    "This object has data for multiple DAQmx scalers, "
                    "use the raw_scaler_data property to get raw data "
                    "for a scale_id")
        return self._raw_data.data

    def read_data(self, offset=0, length=None):
        """ Reads data for this channel from the TDMS file and returns it

            This is for use when the TDMS file was opened without immediately reading all data,
            otherwise the data attribute should be used.

            :param offset: Initial position to read data from.
            :param length: Number of values to attempt to read.
                Fewer values will be returned if attempting to read beyond the end of the available data.
        """
        raw_data = self.tdms_file._read_channel_data(self.path, offset, length)
        return self._scale_data(raw_data)

    @_property_builtin
    def raw_scaler_data(self):
        """ Raw DAQmx scaler data as a dictionary mapping from scale id to raw data arrays
        """
        if self.has_data and self._raw_data is None:
            raise RuntimeError("Channel data has not been read")

        return self._raw_data.scaler_data

    def _scale_data(self, raw_data):
        scale = self._get_scaling()
        if scale is not None:
            return scale.scale(raw_data)
        elif raw_data.scaler_data:
            raise ValueError("Missing scaling information for DAQmx data")
        else:
            return raw_data.data

    def _get_scaling(self):
        try:
            group = self.tdms_file.object(self.group)
        except KeyError:
            group = None
        group_properties = {} if group is None else group.properties
        try:
            root_obj = self.tdms_file.object()
        except KeyError:
            root_obj = None
        file_properties = {} if root_obj is None else root_obj.properties
        return scaling.get_scaling(
            self.properties, group_properties, file_properties)

    def _set_raw_data(self, data):
        self._raw_data = data


def _components_to_path(*args):
    """Convert group and channel to object path"""

    return ('/' + '/'.join(
        ["'" + arg.replace("'", "''") + "'" for arg in args]))


def _hdf_attr_value(value):
    """ Convert a value into a format suitable for an HDF attribute
    """
    if isinstance(value, np.datetime64):
        return np.string_(np.datetime_as_string(value, unit='us', timezone='UTC'))
    return value
